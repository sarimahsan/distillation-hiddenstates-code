import os
import gc
import json
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, TrainingArguments, DataCollatorForSeq2Seq
from huggingface_hub import login

from src.config import HiddenDistillationConfig
from src.models import (
    load_teacher_model,
    load_student_model,
    HiddenStateProjection,
    merge_and_save_model,
    load_distilled_model
)
from src.dataset import InstructionDataset
from src.evaluator import DistillationEvaluator
from src.trainer import HiddenStateDistillationTrainer, TimeBudgetCallback
from src.utils import print_gpu_memory, plot_training_curves

def main():
    parser = argparse.ArgumentParser(description="LLM Hidden-State Distillation Pipeline")
    parser.add_argument("--teacher_model", type=str, default=None, help="Teacher model name/path")
    parser.add_argument("--student_model", type=str, default=None, help="Student model name/path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Per-device train batch size (overrides auto-detection)")
    parser.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps (overrides auto-detection)")
    parser.add_argument("--time_budget", type=float, default=None, help="Time budget in minutes")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the trained models to HF Hub")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token for hub push")
    parser.add_argument("--repo_id_lora", type=str, default=None, help="HF Hub repo ID for the LoRA adapter (e.g. username/repo-lora)")
    parser.add_argument("--repo_id_merged", type=str, default=None, help="HF Hub repo ID for the merged model (e.g. username/repo-merged)")
    
    args = parser.parse_args()

    # Load configuration
    config = HiddenDistillationConfig()
    
    # Override defaults with CLI args
    if args.teacher_model: config.teacher_model_name = args.teacher_model
    if args.student_model: config.student_model_name = args.student_model
    if args.output_dir: config.output_dir = args.output_dir
    if args.num_epochs is not None: config.num_train_epochs = args.num_epochs
    if args.learning_rate is not None: config.learning_rate = args.learning_rate
    if args.time_budget is not None: config.time_budget_minutes = args.time_budget

    # Seed setting
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print("=" * 60)
    print("           LLM Hidden-State Distillation Pipeline           ")
    print("=" * 60)

    # Determine GPU information
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Detected GPU: {gpu_name}  |  Total Memory: {gpu_mem_gb:.1f} GB")
        device = "cuda:0"
    else:
        gpu_mem_gb = 0
        device = "cpu"
        print("WARNING: No GPU detected! Running on CPU.")

    # Determine Batch Size and Grad Accum
    if args.batch_size is not None and args.grad_accum is not None:
        per_device_train_batch_size = args.batch_size
        gradient_accumulation_steps = args.grad_accum
    else:
        if gpu_mem_gb >= 35:
            per_device_train_batch_size, gradient_accumulation_steps = 8, 2
            config.gradient_checkpointing = False
        elif gpu_mem_gb >= 20:
            per_device_train_batch_size, gradient_accumulation_steps = 4, 4
        else:
            per_device_train_batch_size, gradient_accumulation_steps = 2, 8

    print(f"Using Batch Size: {per_device_train_batch_size} | Grad Accum: {gradient_accumulation_steps}")
    print(f"Gradient Checkpointing: {config.gradient_checkpointing}")
    print_gpu_memory("Initial State")

    # Load Tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.student_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print_gpu_memory("After Tokenizer Load")

    # Load Teacher
    print("\n[2/4] Loading teacher model...")
    teacher_model = load_teacher_model(config, device)
    print_gpu_memory("After Teacher Load")

    # Load Student
    print("\n[3/4] Loading student model...")
    student_model = load_student_model(config, device)
    print_gpu_memory("After Student Load")

    # Load Projection Layer
    print("\n[4/4] Creating projection layer...")
    num_layers = 1 if config.match_layers == "last" else student_model.config.num_hidden_layers
    projection = HiddenStateProjection(
        teacher_hidden_size=teacher_model.config.hidden_size,
        student_hidden_size=student_model.config.hidden_size,
        projection_type=config.projection_type,
        num_layers=num_layers,
    ).to(device).to(torch.bfloat16)
    print_gpu_memory("After Projection Layer")

    print("\n" + "="*50 + "\n✓ ALL MODELS AND LAYERS LOADED.\n" + "="*50)

    # Initialize Evaluator & Validation Dataset
    print("\nLoading validation dataset...")
    val_dataset = InstructionDataset(
        config=config, tokenizer=tokenizer, max_length=config.max_seq_length,
        split="val", val_size=config.val_size, seed=config.seed
    )

    evaluator = DistillationEvaluator(
        teacher_model=teacher_model, student_model=student_model, projection=projection,
        tokenizer=tokenizer, config=config, device=device
    )

    # Baseline Evaluation
    print("\n" + "="*60 + "\nBASELINE EVALUATION\n" + "="*60)
    baseline_metrics = evaluator.run_full_evaluation(val_dataset, epoch=-1)
    print("✓ Baseline complete!")

    # Train Dataset Setup
    print("\nLoading training dataset...")
    train_dataset = InstructionDataset(
        config=config, tokenizer=tokenizer, max_length=config.max_seq_length,
        split="train", val_size=config.val_size, seed=config.seed
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True, max_length=config.max_seq_length, return_tensors="pt")

    steps_per_epoch = len(train_dataset) // (per_device_train_batch_size * gradient_accumulation_steps)
    print(f"Train samples: {len(train_dataset)} | Optimizer steps/epoch: {steps_per_epoch} "
          f"| Total planned steps: {steps_per_epoch * config.num_train_epochs}")

    # Training Arguments Setup
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=config.report_to,
        eval_strategy="no",  # We run our own richer evaluation loop
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
    )

    trainer = HiddenStateDistillationTrainer(
        teacher_model=teacher_model,
        projection=projection,
        evaluator=evaluator,
        config=config,
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Custom trainer evaluate() will use val_dataset
        data_collator=data_collator,
        callbacks=[TimeBudgetCallback(config.time_budget_minutes)],
    )

    print("\nStarting training loop...")
    trainer.train()
    print("✓ Training finished!")

    # Save student LoRA adapter and projection weights
    print("\nSaving student model and projection adapters...")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # Post-training Final Evaluation
    print("\n" + "="*60 + "\nFINAL EVALUATION (After Training)\n" + "="*60)
    final_metrics = evaluator.run_full_evaluation(val_dataset, epoch=config.num_train_epochs)

    # Distillation Summary Table
    print("\n" + "="*70)
    print("DISTILLATION SUMMARY")
    print("="*70)
    print(f"\n{'Metric':<35} {'Before':<15} {'After':<15} {'Change':<15}")
    print("-" * 70)
    for name, key, lower_is_better in [
        ("Val Perplexity", "val_perplexity", True),
        ("Teacher-Student KL", "teacher_student_kl", True),
        ("Hidden Cosine Similarity", "cosine_sim_avg", False),
    ]:
        before = baseline_metrics.get(key, float("nan"))
        after = final_metrics.get(key, float("nan"))
        change = after - before
        if not np.isnan(before) and not np.isnan(after):
            symbol = "✓" if (lower_is_better and change < 0) or (not lower_is_better and change > 0) else "✗"
            print(f"{name:<35} {before:<15.4f} {after:<15.4f} {change:+.4f} {symbol}")

    summary_save_path = os.path.join(config.output_dir, "distillation_summary.json")
    with open(summary_save_path, "w") as f:
        json.dump({"baseline": baseline_metrics, "final": final_metrics}, f, indent=2, default=str)
    print(f"\nSaved distillation summary to {summary_save_path}")

    # Plot training/validation curves
    print("\nGenerating training curve plots...")
    curves_plot_path = os.path.join(config.output_dir, "training_curves.png")
    plot_training_curves(config.output_dir, save_path=curves_plot_path)

    # Merge LoRA adapter with base model and save
    merged_output_dir = "./merged_model"
    print(f"\nMerging LoRA adapter with base student model and saving to {merged_output_dir}...")
    
    # Clean up GPU memory of models before merging to avoid OOM
    del teacher_model
    del student_model
    del projection
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        merge_and_save_model(config.student_model_name, config.output_dir, merged_output_dir)
    except Exception as e:
        print(f"WARNING: Model merging failed: {e}. You can merge manually using src.models.merge_and_save_model.")

    # Pushing to Hugging Face Hub (if requested)
    if args.push_to_hub:
        token = args.hf_token or os.environ.get("HF_TOKEN")
        if token:
            login(token=token)
        else:
            print("WARNING: HF token not provided and HF_TOKEN env var is missing. Attempting push with cached credentials...")

        if args.repo_id_lora:
            print(f"Pushing LoRA adapter to Hub: {args.repo_id_lora}...")
            try:
                distilled_student, distilled_tok = load_distilled_model(config.output_dir, config.student_model_name)
                distilled_student.push_to_hub(args.repo_id_lora)
                distilled_tok.push_to_hub(args.repo_id_lora)
                print("✓ LoRA adapter pushed successfully!")
            except Exception as e:
                print(f"Error pushing LoRA adapter: {e}")

        if args.repo_id_merged:
            print(f"Pushing merged model to Hub: {args.repo_id_merged}...")
            try:
                merged_model = AutoModelForCausalLM.from_pretrained(merged_output_dir, trust_remote_code=True)
                merged_tok = AutoTokenizer.from_pretrained(merged_output_dir, trust_remote_code=True)
                merged_model.push_to_hub(args.repo_id_merged)
                merged_tok.push_to_hub(args.repo_id_merged)
                print("✓ Merged model pushed successfully!")
            except Exception as e:
                print(f"Error pushing merged model: {e}")

if __name__ == "__main__":
    main()
