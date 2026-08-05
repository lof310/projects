# **Raw Audio AutoEncoder** -- Work in Progress.

> [!NOTE]
> The internal implementation of the AutoEncoder is intentionally omitted from this public repository. The training setup, a description of the architecture, the notebook used to train it and the results are fully disclosed.

## Architecture
Downsampling using Strided Convolutions of Even kernel size being the double of the stride.
Residual Blocks before the strided convolutions and Dilated Convolutions on the Residual Blocks.
Number of Residual Blocks are specific to each Encoder and Decoder Blocks
SwiGLU in almost all the residual blocks.
The Decoder is not entirely symetric to the Encoder

Downscaling factor in time across depth:
1->2->2->2->2->2->4

Channel dim in Encoder across depth:
1->3->4->6->8->16->24->48->96

Channel dim in Decoder across depth
96->48->24->16->8->6->4->1

Total Downscaling in Time dimension: 128

Total Compression Ratio: 15% <-- In the sense of total number of elements in the output tensor of the encoder relative to the total number of elements in the original input tensor

Total Parameters of The model: 181K

## Training Parameters
```python
seed = 123
epochs = 3
eval_interval = 16 # Interval of steps to evaluate the model
pbar_update_interval = 1 # Interval to update the progress bar
snr_interval = 2 # Interval to Compute the train SI-SNR
batch_size = 32
lr = 6e-4
weight_decay = 1e-2
grad_clip = 3.5
betas = (0.9, 0.995) # Beta hyperparameters for AdamW optimizer
num_threads = 12 # CPU Threads
```

## Dataset
The dataset was created using the script in `stuff/dataset/ADG/dataset_gen.py`.

Sampling Rate: 24Khz
Duration of each sample: 4 seconds
Num Samples: 8320
Min Sources: 1
Max Sources: 2
Peak Norm: 0.95

## Results
The model reached around 10dB Scale-Invariant Signal Noise Ratio (SI-SNR) on only 3 epochs.
Testing it separately on 512 samples of 24Khz single source audio got 10.27dB SI-SNR
And when tested on 512 samples of 48Khz single source audio got 14dB SI-SNR
The model can process 60 seconds of 24Khz raw audio 48x faster than real time on 8 CPU Cores with `powersupersave` mode of the Linux kernel and running at the lowest frequency(around 411Mhz to 1.3GHz), I have a service that automatically shutsdown half of CPU Cores and puts the system into power save mode, when the laptop is disconnected from the charger.

See the Notebook in [`AutoEncoder_Training.ipynb`](AutoEncoder_Training.ipynb)
