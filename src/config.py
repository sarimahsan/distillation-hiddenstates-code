from dataclasses import dataclass, field
from typing import List

@dataclass
class HiddenDistillationConfig:
    """Configuration for Hidden-State Distillation with Full Evaluation"""

    # Model paths
    teacher_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    student_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # Output
    output_dir: str = "./hidden_distill_output"

    # Loss weights: L = ce_loss_weight * L_CE + kd_loss_weight * L_KD + hidden_loss_weight * L_hidden
    ce_loss_weight: float = 0.3
    kd_loss_weight: float = 0.4
    hidden_loss_weight: float = 0.3
    kd_temperature: float = 2.0

    # Hidden state matching
    match_layers: str = "last"        # "last", "all", "sample"
    projection_type: str = "linear"   # "linear", "mlp", "attention"
    normalize_hidden: bool = True

    # LoRA (student only)
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])

    # Quantization (teacher + student both loaded in 4-bit to fit a T4)
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    attn_implementation: str = "sdpa"   # faster than "eager", no extra install needed

    # Training
    num_train_epochs: int = 1
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_seq_length: int = 512           # was 1024 -- halving this roughly doubles+ throughput
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 40                # checkpoint often -- cheap insurance against disconnects
    save_total_limit: int = 2

    # Dataset -- trimmed down + streaming for the big one
    dataset_names: List[str] = field(default_factory=lambda: [
        "tatsu-lab/alpaca",
        "databricks/databricks-dolly-15k",
        "HuggingFaceH4/ultrachat_200k",
    ])
    dataset_splits: List[str] = field(default_factory=lambda: [
        "train",
        "train",
        "train_sft",
    ])
    # Small enough that a T4 training loop finishes comfortably within budget.
    dataset_sizes: List[int] = field(default_factory=lambda: [800, 800, 900])
    # Only ultrachat is huge on disk -- stream it instead of downloading all 200k rows.
    dataset_streaming: List[bool] = field(default_factory=lambda: [False, False, True])

    val_size: int = 120

    # Slower, generation-based eval -- off by default to save time.
    eval_truthfulqa: bool = False
    truthfulqa_sample_size: int = 50

    # Hard wall-clock cap (minutes) for the TRAINING LOOP specifically.
    time_budget_minutes: int = 150

    # Misc
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True
    report_to: str = "none"
