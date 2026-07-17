# projects

Building Towards AGI(or maybe just better LLMs). Work in progress.

## What's in here

- **Orthogonal-Parallel Residuals** - Replaces standard skip connections by splitting sublayer outputs into a parallel component (reinforcement) and an orthogonal component (new information). Learns the mix per layer. At small scale improves validation accuracy only slightly because at those scales (~3M-7M parameters) models are very stable and don't suffer from instability problems. However,the norm of activations stays quite balanced across layers even at small scales. See: [`components/skip-connection`](components/skip-connection)

- **Gradient Conditioning (for SGD)** - A small transformation applied to gradients before the optimizer step. Makes SGD find flatter minima. Gave +7.2-10.2pp percentage point improvement on CIFAR-10 test accuracy in 10 epochs. My goal is to understand why this improvement occurred and how to replicate it at scale with lower cost. See: [`optimization/gradient_conditioning.md`](optimization/gradient_conditioning.md)

- **ShiftMax** - A replacement for Softmax that is more efficient (same FLOPs but no exponentials, so faster in hardware) and has better behavior (no over-confidence). This normalization function is not a replacement for softmax in attention or in loss computation. I plan to use it for components that require normalization for probabilities, good non-linearity and gradient flow, but without over-confidence. See: [`components/shiftmax/README.md`](components/shiftmax/README.md)

- **Early Experiment** - Preliminary architecture from when I was starting. Probably won't include in the first MVP. See: [`stuff/net`](stuff/net)

- **Symbolic CoT Language** - Symbolic language for AI Chain-of-Thought, designed for very small models. See: [`stuff/something.md`](stuff/something.md)

- **Random Character Classification Dataset (RCCD)** - Synthetic Random Character Classification Dataset. See: [`stuff/dataset/RCCD/README.md`](stuff/dataset/RCCD/README.md)

- **Line Intersections Dataset (LID)** - Generates synthetic images of random lines with target labels equal to the number of interior intersection points among the lines. Outputs as either individual PNG files organized by label or a PyTorch tensor pair. See: [`stuff/dataset/LID/dataset_gen.py`](stuff/dataset/LID/dataset_gen.py)

- **Super-Resolution Datatset generator** - A script that generates a dataset for X2 image super-resolution. Scans local images (`.png`, `.jpg`, `.jpeg`) and videos (`.mp4`, `.mkv`) via `ffmpeg`, extracts random crops and generates bicubic LR-HR pairs with various crops per image. See: [`stuff/dataset/SRD/dataset_gen.py`](stuff/dataset/SRD/dataset_gen.py)

- **Audio Dataset Generator** - A script that generates a dataset for training Audio AutoEncoders. See: [`stuff/dataset/ADG/dataset_gen.py`](stuff/dataset/ADG/dataset_gen.py)

- **ColorMixing** - Improved Color Mixing in CNNs. Beats the standard convolutional baseline across all metrics(train/val loss and PSNR). See: [`stuff/colormix/README.md`](stuff/colormix/README.md)

- **Replacement of VGG** - New loss functions that replace the use of VGG for perceptual loss. I cannot make a Benchmark against VGG on CPU, but early results are promising. See: [`stuff/vgg/README.md`](stuff/vgg/README.md)

- **Early Audio Hypothesis Test** - There is a fundamental misalignment in how the field treats Raw Audio Signals. I benchmarked two AutoEncoders, mine and the baseline. Despite having fewer parameters, a smaller receptive field in the time dimension and contrary to the default assumption that uniform temporal processing is optimal for waveform reconstruction, It reaches lower validation loss after 5 epochs. See: [`stuff/audio/hypothesis.md`](stuff/audio/hypothesis.md)

- **Other pieces** - I'm also exploring attention replacements and feed-forward block architectures (complete redesigns, not just new activation functions). Code not published.

## Setup
Everything runs on CPU (my laptop) or my phone (PyTorch on Termux for tiny benchmarks I will not publish here).
