import os
import glob
import random
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F

import matplotlib.pyplot as plt, matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

ROOT = './data'
TARGET_SR = 18432
DURATION = 3.0
NUM_SAMPLES = 1024
MIN_SRC, MAX_SRC = 2, 3
FADE_MS = 5
PEAK_NORM = 0.95
EXTS = ('.mp3','.m4a','.wav')

def probe_files(root, min_sec):
    pool = []
    for ext in EXTS:
        for p in glob.iglob(os.path.join(root, '**', f'*{ext}'), recursive=True):
            try:
                info = torchaudio.info(p)
                if info.num_frames / info.sample_rate >= min_sec:
                    pool.append((p, info.sample_rate, info.num_frames))
            except: pass
    if not pool: raise RuntimeError(f'No audio >= {min_sec}s found in {root}')
    return pool

def mix_sample(pool, sr, dur):
    fade = int(sr * FADE_MS / 1000)
    fi = fo = np.ones(0, dtype=np.float32)
    if fade:
        ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade, dtype=np.float32)))
        fi, fo = ramp, ramp[::-1]
    n = int(sr * dur)
    mix = np.zeros(n, dtype=np.float32)
    for _ in range(random.randint(MIN_SRC, MAX_SRC)):
        path, orig_sr, frames = random.choice(pool)
        start = random.uniform(0, max(0, frames/orig_sr - dur))
        chunk, _ = torchaudio.load(path, frame_offset=int(start*orig_sr), num_frames=int(orig_sr*dur))
        if chunk.size(0) > 1: chunk = chunk.mean(0, keepdim=True)
        chunk = chunk.squeeze(0)
        if orig_sr != sr: chunk = F.resample(chunk, orig_sr, sr)
        seg = chunk.numpy().astype(np.float32)[:n]
        if len(seg) < n: seg = np.pad(seg, (0, n-len(seg)))
        seg *= random.uniform(0.2, 0.8)
        if fade and fade <= n//2:
            seg[:fade] *= fi; seg[-fade:] *= fo
        mix += seg
    peak = np.abs(mix).max()
    if peak > 1e-8: mix *= PEAK_NORM / peak
    return mix

def generate_dataset(num_samples, output_dir='dataset', save_to_file=True, sr=TARGET_SR, dur=DURATION):
    pool = probe_files(ROOT, dur)
    waves = [mix_sample(pool, sr, dur) for _ in tqdm(range(num_samples), desc='Generating Dataset')]
    if save_to_file:
        torch.save(torch.tensor(np.stack(waves), dtype=torch.float32), 'dataset.pt')
    else:
        os.makedirs(output_dir, exist_ok=True)
        for i, w in enumerate(waves):
            torchaudio.save(f'{output_dir}/sample_{i:04d}.wav', torch.from_numpy(w).unsqueeze(0), sr)

def generate_animation(num_samples, sr=TARGET_SR, dur=10.0, fps=2, output='dataset.mp4'):
    pool = probe_files(ROOT, dur)
    waves = [mix_sample(pool, sr, dur) for _ in tqdm(range(num_samples), desc='Generating Animation')]
    fig, ax = plt.subplots(figsize=(12,8))
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    line, = ax.plot(t, waves[0])
    ax.set_ylim(-1.1, 1.1); ax.set_xlabel('Time [s]'); ax.set_ylabel('Amplitude')
    title = ax.set_title('Sample 0')
    def upd(frame):
        line.set_ydata(waves[frame]); title.set_text(f'Sample {frame}'); return line,title
    anim = animation.FuncAnimation(fig, upd, frames=len(waves), interval=10, blit=True)
    anim.save(output, writer=FFMpegWriter(fps=fps, metadata={'title':'Audio Dataset'},
                extra_args=['-threads','32','-preset','fast','-an']), dpi=80)
    plt.close()

generate_dataset(NUM_SAMPLES, save_to_file=True)
#anim = generate_animation(48, dur=0.25, fps=4)
