# Gradient Conditioning: Making SGD Generalize Better With Almost Zero Overhead

**Leinier Orama Fernández**  
`lof310w@gmail.com` | github.com/lof310

---

## Abstract

Vanilla SGD doesn't know which gradient direction(For each weight vector) leads to good generalization. I added a tiny module that scales and shifts the gradient right before the parameter update. On CIFAR‑10, this gave a 7-12% test accuracy improvement over plain SGD, with about 12% extra training time. No new optimizer, no weight decay tuning - just a **hardcoded** but smarter gradient.

---

## 1. The Problem

SGD updates weights:

```
w = w - lr * g
```

`g` is the raw gradient. The optimizer has no clue whether `g` points toward a narrow valley (brittle, poor generalization) or a flat minimum (robust). Grokking, sudden generalization after many epochs, is evidence that SGD gropes around blindly for a long time before stumbling into the right basin. You can add momentum, schedules, weight decay - they help, but they don't change the fact that the gradient signal itself is unprocessed.

I wanted a way to reshape `g` on the fly without reinventing the optimizer. Something cheap, learnable, and removable

---

## 2. Method: Gradient Conditioning

I inserted a small conditioning step between gradient computation and the SGD update:

```
g_cond = scale * g + shift
w = w - lr * g_cond
```

`scale` and `shift` are scalar parameters (one per weight tensor) - not per‑element, just two numbers per tensor. For a typical CNN with ~1M parameters this adds fewer than 100 scalars. These scalars are generated according to the size, length and position in the network of each tensor.

The conditioner does nothing else. No normalization, no momentum inside it, no memory. It's as simple as it gets.

Implementation wraps `torch.optim.SGD` with a pre‑update hook. No special ops, just element‑wise multiply‑add.

---

## 3. Experiment

- **Dataset:** CIFAR‑10 (50k train, 10k test)
- **Models:** Two small CNNs, both ~1M parameters
- **Baseline:** SGD with momentum 0.9, lr = 0.01, no weight decay
- **Conditioned:** Same SGD settings + gradient conditioner
- **Training:** 10 epochs, batch size 32

Everything ran on a laptop CPU.

---

## 4. Results

| Model   | Baseline Acc. | Conditioned Acc. |
|---------|---------------|------------------|
| CNN‑A   | 38.3%         | 45.5%            |
| CNN‑B   | 42.9%         | 53.1%            |

The conditioned models consistently reached higher test accuracy and kept it.
Training time increased by ~12%, entirely due to the extra FLOPs for scaling the gradients. For a million‑parameter model that's negligible.

I don't yet have a clean(mathematical, not heuristic) explanation for why it works so well and how it would be better to translate this improvement into other architectures.
The approach is not a new optimizer. It's a separate, reusable module that can sit on top of any optimizer that uses a gradient. I've only tested it with SGD on vision tasks. But Next I'll Improve it because this is just gradient scaling and try it on language models with AdamW.

---

All public code and configs: [github.com/lof310/projects](https://github.com/lof310/projects)
