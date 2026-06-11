# projects

Experimental replacements for Transformer components. Work in progress.

## What's in here

- **Orthogonal-Parallel Residuals** - Replaces standard skip connections by splitting sublayer outputs into a parallel component (reinforcement) and an orthogonal component (new information). Learns the mix per layer. At small scale improves validation accuracy only by a little because at those small scales around 3M-7M parameters the models are very stable and don't suffer of any unstability problem. But still at small scales the norm of the activations keeps balanced. See: `components/skip-connection/Benchmark_Residual_Stream.ipynb`
- **Gradient Conditioning (for SGD)** - A small transformation applied to gradients before the optimizer step. Makes SGD find flatter minima. Gave 10‑20% improvement on CIFAR‑10. My goal is to find out why such a big improvement happened and how to replicate it at scale with less costs. See `optimization/gradient_conditioning.md`
- **ShiftMax** - A replacement for Softmax that is a little more efficient(Same FLOPs but no exponentials so it's faster in hardware) and has a better behaviour(No over-confidence). This Normalization function is not a replacement for softmax in attention or in the loss computation of a Transformer. But i plan to use it for a component that requires normalization for probabilities, good non-linearity and gradient but also no over-confidence. See `cpmponents/shiftmax`
- **Don't have a name for this** - Something i made when i was starting. I'm probably not going to include this in the first MVP. The second video was the first version with random input. See `stuff/net`
- **Also don't have a name for this** - Symbolic Language for AIs CoT. Made specifically for very small models. See `stuff/something.md`
- **Other pieces** - I'm also poking at attention (replacements of attention) and feed-forward blocks (whole different architectures, not just new activation functions). No published code.

## Setup

Everything runs on CPU (my laptop) or my phone (PyTorch on Termux).

## Why

I think the Transformer is full of things that can be done better. I'm going after them one by one.
