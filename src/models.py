import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
from src.config import HiddenDistillationConfig

class HiddenStateProjection(nn.Module):
    """Projects teacher hidden states to the student's hidden-state dimension."""

    def __init__(self, teacher_hidden_size: int, student_hidden_size: int,
                 projection_type: str = "linear", num_layers: int = 1):
        super().__init__()
        self.teacher_hidden_size = teacher_hidden_size
        self.student_hidden_size = student_hidden_size
        self.projection_type = projection_type

        if projection_type == "linear":
            self.projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(teacher_hidden_size, student_hidden_size, bias=False),
                    nn.LayerNorm(student_hidden_size),
                ) for _ in range(num_layers)
            ])
        elif projection_type == "mlp":
            self.projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(teacher_hidden_size, teacher_hidden_size // 2, bias=False),
                    nn.GELU(),
                    nn.Linear(teacher_hidden_size // 2, student_hidden_size, bias=False),
                    nn.LayerNorm(student_hidden_size),
                ) for _ in range(num_layers)
            ])
        elif projection_type == "attention":
            self.projections = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=student_hidden_size, num_heads=8,
                    kdim=teacher_hidden_size, vdim=teacher_hidden_size, batch_first=True,
                ) for _ in range(num_layers)
            ])
            self.query_proj = nn.Linear(student_hidden_size, student_hidden_size, bias=False)
            self.out_norm = nn.LayerNorm(student_hidden_size)
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")

    def forward(self, teacher_hidden, layer_idx: int = 0, student_hidden=None):
        proj = self.projections[layer_idx]
        if self.projection_type == "attention":
            assert student_hidden is not None, "student_hidden required for attention projection"
            query = self.query_proj(student_hidden)
            projected, _ = proj(query, teacher_hidden, teacher_hidden)
            projected = self.out_norm(projected)
        else:
            projected = proj(teacher_hidden)
        return projected


def get_bnb_config(config: HiddenDistillationConfig) -> BitsAndBytesConfig:
    """Creates a BitsAndBytesConfig for 4-bit model loading."""
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )


def load_teacher_model(config: HiddenDistillationConfig, device: str = "cuda") -> AutoModelForCausalLM:
    """Loads the teacher model in a quantized format and sets it to evaluation mode."""
    bnb_config = get_bnb_config(config)
    print(f"Loading Teacher Model: {config.teacher_model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device} if device else "auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
        attn_implementation=config.attn_implementation,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_student_model(config: HiddenDistillationConfig, device: str = "cuda") -> get_peft_model:
    """Loads the student model, prepares it for 4-bit training, and wraps it with LoRA."""
    bnb_config = get_bnb_config(config)
    print(f"Loading Student Model: {config.student_model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.student_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device} if device else "auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
        attn_implementation=config.attn_implementation,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    return model


def merge_and_save_model(base_model_name: str, lora_path: str, save_path: str):
    """Merges the trained LoRA adapter into the base model and saves the merged weights."""
    print(f"Loading base student model to merge: {base_model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    print(f"Loading LoRA adapter from {lora_path} and merging...")
    merged_model = PeftModel.from_pretrained(base_model, lora_path)
    merged_model = merged_model.merge_and_unload()

    print(f"Saving merged model to {save_path}...")
    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Merge and save completed!")


def load_distilled_model(output_dir: str, base_model_name: str):
    """Loads the distilled student model (base + LoRA adapter) for inference."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, output_dir)
    model.eval()
    return model, tokenizer
