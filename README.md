# projects

Experimental replacements for Transformer components. Work in progress.

## What's in here

- **Orthogonal-Parallel Residuals** - Replaces standard skip connections by splitting sublayer outputs into a parallel component (reinforcement) and an orthogonal component (new information). Learns the mix per layer. At small scale improves validation accuracy only slightly because at those scales (~3M-7M parameters) models are very stable and don't suffer from instability problems. However,the norm of activations stays quite balanced across layers even at small scales. See: `components/skip-connection/Benchmark_Residual_Stream.ipynb`

- **Gradient Conditioning (for SGD)** - A small transformation applied to gradients before the optimizer step. Makes SGD find flatter minima. Gave +7.2-10.2pp percentage point improvement on CIFAR-10 test accuracy in 10 epochs. My goal is to understand why this improvement occurred and how to replicate it at scale with lower cost. See: `optimization/gradient_conditioning.md`

- **ShiftMax** - A replacement for Softmax that is more efficient (same FLOPs but no exponentials, so faster in hardware) and has better behavior (no over-confidence). This normalization function is not a replacement for softmax in attention or in loss computation. I plan to use it for components that require normalization for probabilities, good non-linearity and gradient flow, but without over-confidence. See: `components/shiftmax`

- **Early Experiment** - Preliminary architecture from when I was starting. Probably won't include in the first MVP. See: `stuff/net`

- **Symbolic CoT Language** - Symbolic language for AI Chain-of-Thought, designed for very small models. See: `stuff/something.md`

- **Other pieces** - I'm also exploring attention replacements and feed-forward block architectures (complete redesigns, not just new activation functions). Code not published.

## Setup
Everything runs on CPU (my laptop) or my phone (PyTorch on Termux).

## Why

I think the Transformer has components that can be improved. I'm addressing them 
one by one.