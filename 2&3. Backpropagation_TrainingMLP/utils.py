"""
Utilities for MLP Analysis (3.3 & 3.4)
=======================================
Contains diagnostic probes, plotting functions, and weight visualization
tools to support the main training script.
"""
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- Diagnostic Probes ---

@torch.no_grad()
def activation_deadness(model, x):
    """
    ## What is this? A Health Check for Neurons 🩺
    This function acts like a stethoscope for our network. It uses PyTorch "hooks"
    to listen to the output of each ReLU activation.

    ## Why do we care? The "Dying ReLU" Problem
    A ReLU neuron "dies" if it always outputs 0. If it's always 0, its gradient
    is always 0, and it stops learning entirely. This is a common problem in deep networks.

    ## What to look for:
    - `zeros_frac`: The fraction of outputs that are zero. If this is high (e.g., > 0.9),
      the layer is mostly "dead" and not contributing much.
    - `mean`/`std`: Healthy layers should have activations with a non-zero mean and standard deviation.
    """
    model.eval()
    activations = []
    hooks = []

    def make_hook(name):
        def hook_fn(_, __, output):
            mean = output.mean().item()
            std = output.std().item()
            zeros_frac = (output == 0).float().mean().item()
            activations.append({"name": name, "mean": mean, "std": std, "zeros_frac": zeros_frac})
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Move input tensor to the same device as the model's parameters
    model(x.to(next(model.parameters()).device))
    for hook in hooks:
        hook.remove()
    return activations


def grad_norms(model, criterion, x, y, device):
    """
    ## What is this? A "Learning Signal" Strength Meter 📶
    This function measures the magnitude (L2 norm) of the gradients for each
    layer's weights after a single backward pass.

    ## Why do we care? Vanishing & Exploding Gradients
    The gradient is the "learning signal" sent from the loss function back through
    the network.
    - **Vanishing Gradients**: If the signal is too weak (gradient norm is tiny, e.g., 1e-8),
      the early layers of the network learn extremely slowly or not at all.
    - **Exploding Gradients**: If the signal is too strong (norm is huge), training
      becomes unstable.

    ## What to look for:
    A healthy network should have reasonably-sized gradient norms across all layers.
    There shouldn't be a dramatic drop-off from the last layer to the first.
    """
    model.train()
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            norms.append({"name": name, "norm": p.grad.norm().item()})

    model.zero_grad()
    return norms

# --- Plotting & Visualization ---

# Helper function for weight visualizations
def collect_layer_weights(model: nn.Module):
    weights = {}
    # Check if model has a 'layers' attribute of type ModuleList
    if hasattr(model, 'layers') and isinstance(model.layers, nn.ModuleList):
        idx_hidden = 1
        for i, m in enumerate(model.layers):
            if isinstance(m, nn.Linear):
                key = f"hidden_{idx_hidden}" if i < len(model.layers)-1 else "output"
                weights[key] = m.weight.detach().cpu().numpy()
                idx_hidden += 1
    return weights

def plot_val_curves(results, title="Validation Accuracy"):
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        plt.plot(data["history"]["val_acc"], label=f"{name} ({data['history']['val_acc'][-1]:.3f})")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def visualize_first_layer_filters(model: nn.Module, max_filters=16):
    weights = collect_layer_weights(model)
    if "hidden_1" not in weights:
        print("Could not find 'hidden_1' weights to visualize.")
        return

    w = weights["hidden_1"]  # (H, 784)
    k = min(max_filters, w.shape[0])
    cols = int(math.sqrt(k)); rows = math.ceil(k/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(k):
        r, c = divmod(i, cols)
        filt = w[i].reshape(28, 28)
        vmax = np.abs(filt).max() + 1e-12
        im = axes[r, c].imshow(filt, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[r, c].set_title(f"Filter {i}", fontsize=9)
        axes[r, c].axis("off")
        plt.colorbar(im, ax=axes[r, c], shrink=0.6)
    fig.suptitle("First-layer filters (flattened image → features)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def weight_distributions(model: nn.Module):
    weights = collect_layer_weights(model)
    if not weights:
        print("Could not collect weights to visualize distributions.")
        return

    L = len(weights)
    cols = 2; rows = math.ceil(L/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows), squeeze=False)
    axes = axes.flatten()
    for i, (name, W) in enumerate(weights.items()):
        ax = axes[i]
        flat = W.ravel()
        ax.hist(flat, bins=50, color="#3b82f6", alpha=0.8, edgecolor="black", linewidth=0.4)
        mu, sd = flat.mean(), flat.std()
        ax.axvline(mu, color="red", linestyle="--", label=f"μ={mu:.3f}")
        ax.axvline(mu+sd, color="orange", linestyle=":", alpha=0.8)
        ax.axvline(mu-sd, color="orange", linestyle=":", alpha=0.8, label=f"σ={sd:.3f}")
        ax.set_title(name); ax.legend(); ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Weight distributions across layers")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- Wrapper Functions to Simplify Main Script ---

def run_all_diagnostics(model, criterion, xb, yb, device):
    """A wrapper to run and print all diagnostic tools."""
    deadness = activation_deadness(model, xb)
    print("Activation Health (Zeros %):")
    for act in deadness:
        print(f"  - {act['name']}: {act['zeros_frac']:.1%}")

    # This ensures the 'device' argument is passed to grad_norms
    norms = grad_norms(model, criterion, xb, yb, device)
    print("\nGradient Signal Strength (L2 Norm):")
    for gn in norms:
        print(f"  - {gn['name']:<25}: {gn['norm']:.3e}")

def run_all_visualizations(model):
    """A wrapper to run all weight visualization plots."""
    print("Visualizing model weights...")
    visualize_first_layer_filters(model)
    weight_distributions(model)
