# projects
cool stuff

# projects

Experimental replacements for Transformer components. Work in progress.

## What's in here

- **Orthogonal-Parallel Residuals** – Replaces standard skip connections by splitting sublayer outputs into a parallel component (reinforcement) and an orthogonal component (new information). Learns the mix per layer. At small scale improves validation accuracy only by a little because at those small scales around 3M-7M parameters the models are very stable and don't suffer of any stability problem. But still at small scales the norm of the activations keeps balanced. See: `components/Benchmark_Residual_Stream.ipynb`
- **Gradient Conditioning (for SGD)** – A small transformation applied to gradients before the optimizer step. Makes SGD find flatter minima. Gave 10‑20% improvement on CIFAR‑10. My goal is to find out why such a big improvement happened and how to replicate it at scale with less costs. See `optimization/gradient_conditioning.md`
- **Other pieces** – I'm also poking at attention (replacements of attention) and feed-forward blocks (whole different architectures, not just new activation functions). No published code yet.

## Setup

Everything runs on CPU (my laptop) or my phone (PyTorch on Termux).

## Why

I think the Transformer is full of things that can be done better. I'm going after them one by one.
