import os
import json
import torch
from typing import Optional

def print_gpu_memory(stage: str = ""):
    """Prints the currently allocated and reserved GPU memory in GB."""
    if not torch.cuda.is_available():
        print(f"[GPU Memory {stage}]: no CUDA device")
        return
    allocated = torch.cuda.memory_allocated(0) / 1e9
    cached = torch.cuda.memory_reserved(0) / 1e9
    print(f"[GPU Memory {stage}]: Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")


def plot_training_curves(output_dir: str, save_path: Optional[str] = None):
    """Plots training/validation metrics stored in epoch_metrics.json and saves/shows the plot."""
    # Try importing matplotlib safely (could be run in headless environments)
    import matplotlib
    # Use Agg back-end to avoid requiring a display when running scripts
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_path = os.path.join(output_dir, "epoch_metrics.json")
    if not os.path.exists(metrics_path):
        print(f"No metrics file found at {metrics_path}")
        return
    
    with open(metrics_path) as f:
        metrics = json.load(f)

    if not metrics:
        print("Metrics file is empty.")
        return

    epochs = [m.get("epoch", i) for i, m in enumerate(metrics)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    def plot_metric(ax, key, title):
        vals = [m.get(key) for m in metrics if key in m]
        if vals:
            ax.plot(epochs[:len(vals)], vals, marker="o", color="#4F46E5", linewidth=2)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.6)

    plot_metric(axes[0][0], "val_perplexity", "Validation Perplexity")
    plot_metric(axes[0][1], "teacher_student_kl", "Teacher-Student KL")
    plot_metric(axes[1][0], "cosine_sim_avg", "Hidden Cosine Similarity")
    plot_metric(axes[1][1], "val_loss", "Validation Loss")
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved training curves to {save_path}")
    else:
        default_save = os.path.join(output_dir, "training_curves.png")
        plt.savefig(default_save, dpi=300)
        print(f"Saved training curves to {default_save}")
        
    plt.close()
