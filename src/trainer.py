import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainerCallback
from typing import List, Dict, Tuple
from src.config import HiddenDistillationConfig
from src.evaluator import DistillationEvaluator

class TimeBudgetCallback(TrainerCallback):
    """Stops training cleanly once `budget_minutes` of wall-clock time has elapsed.
    A checkpoint has already been written by save_strategy="steps" by that point, so
    stopping here loses at most `save_steps` worth of progress -- not the whole run."""

    def __init__(self, budget_minutes: float):
        self.budget_seconds = budget_minutes * 60
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"[TimeBudget] Training loop budget: {self.budget_seconds/60:.0f} minutes.")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        if elapsed > self.budget_seconds:
            print(f"[TimeBudget] Elapsed {elapsed/60:.1f} min >= budget "
                  f"{self.budget_seconds/60:.0f} min. Stopping training and saving.")
            control.should_training_stop = True
            control.should_save = True
        return control


class HiddenStateDistillationTrainer(Trainer):
    """Custom trainer:
    - L = ce_loss_weight * L_CE + kd_loss_weight * L_KD + hidden_loss_weight * L_hidden
    - Custom evaluate() runs the full DistillationEvaluator suite
    - Saves the projection head + per-eval metrics alongside the LoRA adapter
    """

    def __init__(self, teacher_model: nn.Module, projection: nn.Module,
                 evaluator: DistillationEvaluator, config: HiddenDistillationConfig,
                 *args, **kwargs):
        self.val_dataset = kwargs.get("eval_dataset", None)
        super().__init__(*args, **kwargs)

        self.teacher_model = teacher_model
        self.projection = projection
        self.evaluator = evaluator
        self.distill_config = config

        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False

        self.layer_pairs = self._get_layer_pairs()
        self.epoch_metrics = []

    def _get_layer_pairs(self) -> List[Tuple[int, int]]:
        t_layers = self.teacher_model.config.num_hidden_layers
        s_layers = self.model.config.num_hidden_layers
        match_config = self.distill_config.match_layers
        if match_config == "last":
            return [(t_layers - 1, s_layers - 1)]
        elif match_config == "all":
            return [(int(t), s) for s, t in enumerate(np.linspace(0, t_layers - 1, s_layers))]
        elif match_config == "sample":
            n = min(4, s_layers)
            s_idx = np.linspace(0, s_layers - 1, n, dtype=int)
            t_idx = np.linspace(0, t_layers - 1, n, dtype=int)
            return list(zip(t_idx, s_idx))
        return [(t_layers - 1, s_layers - 1)]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids, attention_mask, labels = inputs["input_ids"], inputs["attention_mask"], inputs["labels"]

        student_outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                 labels=None, output_hidden_states=True, return_dict=True)

        shift_logits = student_outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss_fct = nn.CrossEntropyLoss(reduction="none")
        ce_loss_per_token = ce_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        mask = (shift_labels != -100).float()
        ce_loss = (ce_loss_per_token * mask.view(-1)).sum() / mask.sum()

        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask,
                                                 output_hidden_states=True, return_dict=True)

        kd_loss = torch.tensor(0.0, device=input_ids.device)
        if self.distill_config.kd_loss_weight > 0:
            T = self.distill_config.kd_temperature
            shift_student_logits = student_outputs.logits[..., :-1, :].contiguous() / T
            shift_teacher_logits = teacher_outputs.logits[..., :-1, :].contiguous() / T
            student_log_probs = F.log_softmax(shift_student_logits, dim=-1)
            teacher_probs = F.softmax(shift_teacher_logits, dim=-1)
            per_token_kd = F.kl_div(
                student_log_probs.view(-1, student_log_probs.size(-1)),
                teacher_probs.view(-1, teacher_probs.size(-1)),
                reduction="none",
            ).sum(dim=-1) * (T ** 2)
            kd_loss = (per_token_kd * mask.view(-1)).sum() / mask.sum()

        hidden_loss = torch.tensor(0.0, device=input_ids.device)
        avg_cosine_sim = 0.0
        if self.distill_config.hidden_loss_weight > 0:
            total_hidden_loss, total_cosine_sim, num_layers = 0.0, 0.0, 0
            for layer_idx, (t_idx, s_idx) in enumerate(self.layer_pairs):
                teacher_hidden = teacher_outputs.hidden_states[t_idx + 1]
                student_hidden = student_outputs.hidden_states[s_idx + 1]
                projected = (self.projection(teacher_hidden, layer_idx, student_hidden)
                             if self.distill_config.projection_type == "attention"
                             else self.projection(teacher_hidden, layer_idx))

                if self.distill_config.normalize_hidden:
                    proj_norm, stud_norm = F.normalize(projected, dim=-1), F.normalize(student_hidden, dim=-1)
                    layer_loss = ((proj_norm - stud_norm) ** 2).mean()
                    cos_sim = F.cosine_similarity(proj_norm, stud_norm, dim=-1).mean()
                else:
                    hidden_mask = attention_mask.unsqueeze(-1).float()
                    layer_loss = ((projected - student_hidden) ** 2 * hidden_mask).sum() / (
                        hidden_mask.sum() * projected.size(-1))
                    cos_sim = F.cosine_similarity(projected, student_hidden, dim=-1).mean()
                total_hidden_loss += layer_loss
                total_cosine_sim += cos_sim.item()
                num_layers += 1
            if num_layers > 0:
                hidden_loss = total_hidden_loss / num_layers
                avg_cosine_sim = total_cosine_sim / num_layers

        total_loss = (self.distill_config.ce_loss_weight * ce_loss
                      + self.distill_config.kd_loss_weight * kd_loss
                      + self.distill_config.hidden_loss_weight * hidden_loss)

        if self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0:
            self.log({"loss/ce": ce_loss.item(), "loss/kd": kd_loss.item(),
                      "loss/hidden": hidden_loss.item(), "loss/total": total_loss.item(),
                      "train/cosine_sim": avg_cosine_sim})

        return (total_loss, student_outputs) if return_outputs else total_loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_results = self.evaluator.run_full_evaluation(self.val_dataset, epoch=int(self.state.epoch or 0))
        prefixed = {f"eval/{k}": v for k, v in eval_results.items() if k != "epoch"}
        self.log(prefixed)
        self.epoch_metrics.append(eval_results)
        return prefixed

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir if output_dir else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        super().save_model(output_dir, _internal_call)
        
        # Save the projection head weights
        torch.save(self.projection.state_dict(), os.path.join(output_dir, "hidden_projection.pt"))
        # Save validation evaluation metrics
        with open(os.path.join(output_dir, "epoch_metrics.json"), "w") as f:
            json.dump(self.epoch_metrics, f, indent=2)
        print(f"Saved model, projection, and metrics to {output_dir}")
