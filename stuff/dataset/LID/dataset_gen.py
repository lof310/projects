import random
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from scipy.ndimage import map_coordinates, gaussian_filter
from tqdm import tqdm
import numpy as np

import torch

def intersect_point(A, B, C, D):
    denom = (B[0]-A[0])*(D[1]-C[1]) - (B[1]-A[1])*(D[0]-C[0])
    if abs(denom) < 1e-9:
        return None
    t = ((C[0]-A[0])*(D[1]-C[1]) - (C[1]-A[1])*(D[0]-C[0])) / denom
    u = ((C[0]-A[0])*(B[1]-A[1]) - (C[1]-A[1])*(B[0]-A[0])) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (A[0] + t*(B[0]-A[0]), A[1] + t*(B[1]-A[1]))
    return None

def is_endpoint(p, A, B):
    return np.hypot(p[0]-A[0], p[1]-A[1]) < 1e-9 or np.hypot(p[0]-B[0], p[1]-B[1]) < 1e-9

def generate_image(size):
    num_lines = random.randint(2, 6)

    min_gap, min_intersect_dist, thickness_max, blur_max = (
        max(1, int(size * 0.05)),
        max(1.0, size * 0.025),
        max(1, int(size * 0.02)),
        min(2.0, size * 0.06)
    )

    diag_gap = min_gap * np.sqrt(2)

    H, V = set(range(size)), set(range(size))
    D1, D2 = set(range(-size+2, size-1)), set(range(1, (2*size)-2))

    if num_lines > len(H) + len(V) + len(D1) + len(D2):
        raise ValueError(f"Cannot place {num_lines} lines in {size} image with min_gap={min_gap}")

    lines = []
    orientations = ['H', 'V', 'D1', 'D2']
    for _ in range(num_lines):
        orient = random.choice(orientations)

        if orient == 'H':
            y = random.choice(list(H))
            H = {v for v in H if abs(v - y) >= min_gap}
            A, B = (0, y), (size-1, y)
        elif orient == 'V':
            x = random.choice(list(V))
            V = {v for v in V if abs(v - x) >= min_gap}
            A, B = (x, 0), (x, size-1)
        elif orient == 'D1':
            b = random.choice(list(D1))
            D1 = {v for v in D1 if abs(v - b) >= diag_gap}
            A, B = (max(0, -b), max(0, b)), (min(size-1, size-1-b), min(size-1, size-1+b))
        else:
            c = random.choice(list(D2))
            D2 = {v for v in D2 if abs(v - c) >= diag_gap}
            A, B = (max(0, c-(size-1)), min(size-1, c)), (min(size-1, c), max(0, c-(size-1)))

        lines.append((A, B))

    points = []
    for i, (A, B) in enumerate(lines):
        for j in range(i+1, len(lines)):
            C, D = lines[j]
            p = intersect_point(A, B, C, D)
            if p is None or is_endpoint(p, A, B) or is_endpoint(p, C, D):
                continue
            if all(np.hypot(p[0]-q[0], p[1]-q[1]) >= min_intersect_dist for q in points):
                points.append(p)

    label = min(len(points), ((num_lines**2)-num_lines)/2)

    img = Image.new('L', (size, size), color=255)
    draw = ImageDraw.Draw(img)
    thickness = random.randint(1, thickness_max+1)

    # Generate Random Noise
    noise = random.uniform(0, 0.1) * np.random.randn(size, size)
    noise = np.clip(noise * 255, 0, 255)

    # Draw Lines
    for A, B in lines:
        draw.line([A, B], fill=0, width=thickness)

    # Apply Gaussian Blur
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, blur_max)))

    # Apply Noise
    img_array = np.array(img, dtype=np.float32)

    return np.clip((img_array-noise), 0, 255).astype(np.uint8), label

def generate_dataset(num_samples=10000, output_dir='dataset', save_to_file=True, size=128):
    if not save_to_file: os.makedirs(output_dir, exist_ok=True)

    pbar = tqdm(total=num_samples, desc="Generating Dataset")

    inputs, targets = [], []

    for sample in range(num_samples):
        img_array, label = generate_image(size)

        # Save image
        if save_to_file:
            inputs.append(img_array)
            targets.append(label)
        else:
            label_dir = os.path.join(output_dir, f"{label}"); os.makedirs(label_dir, exist_ok=True)
            Image.fromarray(img_array).save(os.path.join(label_dir, f'{label}_{sample:04d}.png'))

        pbar.update(1)

    if save_to_file:
        inputs = torch.tensor(np.array(inputs), dtype=torch.uint8)
        targets = torch.tensor(targets, dtype=torch.long)
        torch.save((inputs, targets), 'dataset.pt')

def generate_animation(characters, fonts, special_characters, special_fonts, num_samples=100, size=64):
    pbar = tqdm(total=num_samples, desc="Generating Animation")

    images = []

    for sample in range(num_samples):
        img = generate_image(size)[0]
        images.append(img)
        pbar.update(1)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(f"Dataset Images {len(images)}", fontsize=16)
    fig.set_dpi(40)

    img = ax.imshow(images[0], cmap='gray')
    title = ax.set_title(f"Image 0")
    plt.colorbar(img, ax=ax)

    def update(frame):
        img.set_data(images[frame])
        title.set_text(f"Image {frame}")
        return [img, title]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(images),
        interval=10,
        blit=True
    )

    return anim, fig, len(images)

generate_dataset(num_samples=20000, save_to_file=True, size=144)

#anim, fig, images = generate_animation(characters, fonts, special_characters, special_fonts, num_samples_per_char=48, size=(64, 64))
#writer = FFMpegWriter(
#    fps=24,
#    metadata={'title': 'Dataset Images', 'artist': 'Python Matplotlib'},
#    extra_args=['-threads', '16', '-preset', 'fast', '-an']
#)

#pbar = tqdm(total=images, desc="Saving Animation")

#output_filename = 'dataset.mp4'
#anim.save(output_filename, writer=writer, progress_callback=lambda i, n: pbar.update(1))
#plt.close(fig)
