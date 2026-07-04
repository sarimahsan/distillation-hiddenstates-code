import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
from typing import List, Dict, Tuple
from src.config import HiddenDistillationConfig

class DistillationEvaluator:
    """Validation perplexity, teacher-student KL, hidden-state cosine similarity,
    and (optionally) TruthfulQA."""

    def __init__(self, teacher_model: nn.Module, student_model: nn.Module, projection: nn.Module,
                 tokenizer, config: HiddenDistillationConfig, device: str = "cuda"):
        self.teacher = teacher_model
        self.student = student_model
        self.projection = projection
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.layer_pairs = self._get_layer_pairs()

    def _get_layer_pairs(self) -> List[Tuple[int, int]]:
        t_layers = self.teacher.config.num_hidden_layers
        s_layers = self.student.config.num_hidden_layers
        if self.config.match_layers == "last":
            return [(t_layers - 1, s_layers - 1)]
        elif self.config.match_layers == "all":
            return [(int(t), s) for s, t in enumerate(np.linspace(0, t_layers - 1, s_layers))]
        elif self.config.match_layers == "sample":
            n = min(4, s_layers)
            s_idx = np.linspace(0, s_layers - 1, n, dtype=int)
            t_idx = np.linspace(0, t_layers - 1, n, dtype=int)
            return list(zip(t_idx, s_idx))
        return [(t_layers - 1, s_layers - 1)]

    def _make_loader(self, val_dataset, batch_size: int) -> DataLoader:
        return DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=DataCollatorForSeq2Seq(
                self.tokenizer, padding=True, max_length=self.config.max_seq_length, return_tensors="pt"
            ),
        )

    def compute_validation_perplexity(self, val_dataset, batch_size: int = 4) -> Dict[str, float]:
        self.student.eval()
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            for batch in self._make_loader(val_dataset, batch_size):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.student(**batch)
                mask = (batch["labels"] != -100).float()
                n = mask.sum().item()
                total_loss += outputs.loss.item() * n
                total_tokens += n
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        return {"val_loss": avg_loss, "val_perplexity": float(np.exp(avg_loss))}

    def compute_teacher_kl_divergence(self, val_dataset, batch_size: int = 2) -> Dict[str, float]:
        self.student.eval()
        self.teacher.eval()
        total_kl, total_tokens = 0.0, 0
        student_vocab_size = self.student.config.vocab_size
        with torch.no_grad():
            for batch in self._make_loader(val_dataset, batch_size):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]

                teacher_outputs = self.teacher(input_ids=input_ids, attention_mask=attention_mask)
                student_outputs = self.student(input_ids=input_ids, attention_mask=attention_mask)

                T = self.config.kd_temperature
                teacher_logits = teacher_outputs.logits[..., :student_vocab_size]  # guard vocab mismatch
                teacher_probs = F.softmax(teacher_logits / T, dim=-1)
                student_log_probs = F.log_softmax(student_outputs.logits / T, dim=-1)

                kl_per_token = F.kl_div(
                    student_log_probs.view(-1, student_log_probs.size(-1)),
                    teacher_probs.view(-1, teacher_probs.size(-1)),
                    reduction="none",
                ).sum(dim=-1) * (T ** 2)

                mask = (labels != -100).float().view(-1)
                total_kl += (kl_per_token * mask).sum().item()
                total_tokens += mask.sum().item()
        avg_kl = total_kl / total_tokens if total_tokens > 0 else float("inf")
        return {"teacher_student_kl": avg_kl}

    def compute_hidden_cosine_similarity(self, val_dataset, max_batches: int = 10) -> Dict[str, float]:
        self.student.eval()
        self.teacher.eval()
        cosine_sims = {s_idx: [] for _, s_idx in self.layer_pairs}
        with torch.no_grad():
            for batch_idx, batch in enumerate(self._make_loader(val_dataset, batch_size=2)):
                if batch_idx >= max_batches:
                    break
                batch = {k: v.to(self.device) for k, v in batch.items()}
                input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]

                teacher_outputs = self.teacher(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                student_outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

                for layer_idx, (t_idx, s_idx) in enumerate(self.layer_pairs):
                    teacher_hidden = teacher_outputs.hidden_states[t_idx + 1]
                    student_hidden = student_outputs.hidden_states[s_idx + 1]
                    projected = (self.projection(teacher_hidden, layer_idx, student_hidden)
                                 if self.config.projection_type == "attention"
                                 else self.projection(teacher_hidden, layer_idx))
                    projected = F.normalize(projected, dim=-1)
                    student_hidden = F.normalize(student_hidden, dim=-1)
                    cos_sim = F.cosine_similarity(projected, student_hidden, dim=-1)
                    mask = attention_mask.bool()
                    cosine_sims[s_idx].extend(cos_sim[mask].cpu().tolist())
        results = {f"cosine_sim_layer_{s_idx}": float(np.mean(sims)) for s_idx, sims in cosine_sims.items() if sims}
        if results:
            results["cosine_sim_avg"] = float(np.mean(list(results.values())))
        return results

    def evaluate_truthfulqa(self, sample_size: int = 50) -> Dict[str, float]:
        try:
            tq = load_dataset("truthfulqa/truthfulqa", "multiple_choice", split="validation")
        except Exception as e:
            print(f"Could not load TruthfulQA: {e}")
            return {"truthfulqa_error": str(e)}

        indices = np.random.choice(len(tq), min(sample_size, len(tq)), replace=False)
        teacher_correct = student_correct = total = 0
        for idx in indices:
            question = tq[int(idx)]["question"]
            choices = tq[int(idx)]["mc1_targets"]["choices"]
            labels = tq[int(idx)]["mc1_targets"]["labels"]
            correct_answer = choices[labels.index(1)]

            teacher_answer = self._generate_answer(self.teacher, question)
            student_answer = self._generate_answer(self.student, question)
            if correct_answer.lower() in teacher_answer.lower():
                teacher_correct += 1
            if correct_answer.lower() in student_answer.lower():
                student_correct += 1
            total += 1

        teacher_acc = teacher_correct / total if total else 0
        student_acc = student_correct / total if total else 0
        return {
            "truthfulqa_teacher_acc": teacher_acc,
            "truthfulqa_student_acc": student_acc,
            "truthfulqa_hallucination_gap": student_acc - teacher_acc,
            "truthfulqa_samples": float(total),
        }

    def _generate_answer(self, model, question: str, max_new_tokens: int = 50) -> str:
        messages = [{"role": "user", "content": question}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    def run_full_evaluation(self, val_dataset, epoch: int = 0) -> Dict[str, float]:
        print(f"\n{'='*50}\nRunning Evaluation - Epoch {epoch}\n{'='*50}")
        results = {"epoch": float(epoch)}

        print("Computing validation perplexity...")
        results.update(self.compute_validation_perplexity(val_dataset))
        print(f"  Val Loss: {results['val_loss']:.4f}  |  Val Perplexity: {results['val_perplexity']:.2f}")

        print("Computing teacher-student KL divergence...")
        results.update(self.compute_teacher_kl_divergence(val_dataset))
        print(f"  KL(T||S): {results['teacher_student_kl']:.4f}")

        print("Computing hidden state cosine similarity...")
        results.update(self.compute_hidden_cosine_similarity(val_dataset))
        if "cosine_sim_avg" in results:
            print(f"  Avg Cosine Sim: {results['cosine_sim_avg']:.4f}")

        if self.config.eval_truthfulqa:
            print(f"Computing TruthfulQA (n={self.config.truthfulqa_sample_size})...")
            results.update(self.evaluate_truthfulqa(self.config.truthfulqa_sample_size))
            if "truthfulqa_teacher_acc" in results:
                print(f"  Teacher Acc: {results['truthfulqa_teacher_acc']:.2%}  |  "
                      f"Student Acc: {results['truthfulqa_student_acc']:.2%}  |  "
                      f"Gap: {results['truthfulqa_hallucination_gap']:.2%}")

        print(f"{'='*50}\n")
        return results
