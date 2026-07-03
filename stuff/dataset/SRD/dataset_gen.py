import os
import glob
import random
import subprocess
import tempfile

from PIL import Image

import torch
from torchvision import transforms

from tqdm import tqdm

ROOT_DIR = './data'
OUTPUT_FILE = "dataset.pt"
CROP_MIN = 32
CROP_MAX = 1024
TARGET_SIZE = 92
FRAMES_PER_VIDEO = 128
FFMPEG_PATH = "ffmpeg"
FFPROBE_PATH = "ffprobe"
IMG_EXTS = ('.png', '.jpg', '.jpeg')
VID_EXTS = ('.mp4', '.mkv')
SAMPLE_WEIGHTS = [1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8]

to_tensor = transforms.ToTensor()

def get_video_duration(path):
    cmd = [FFPROBE_PATH, '-v', 'error', '-show_entries', 'format=duration',
           '-of', 'default=noprint_wrappers=1:nokey=1', path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip())

def get_video_frame(path, time):
    cmd = [FFMPEG_PATH, '-ss', str(time), '-i', path, '-vframes', '1',
           '-f', 'image2pipe', '-']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {err.decode()}")
    return Image.open(tempfile.SpooledTemporaryFile()).open(out).convert("RGB")

image_paths = []
for ext in IMG_EXTS:
    image_paths.extend(glob.glob(os.path.join(ROOT_DIR, '**', f'*{ext}'), recursive=True))
video_paths = []
for ext in VID_EXTS:
    video_paths.extend(glob.glob(os.path.join(ROOT_DIR, '**', f'*{ext}'), recursive=True))

inputs = []
targets = []

def process_image(img):
    w, h = img.size
    n_crops = random.choice(SAMPLE_WEIGHTS)
    for _ in range(n_crops):
        crop_h = random.randrange(CROP_MIN, min(h, CROP_MAX) + 1, 4)
        crop_w = random.randrange(CROP_MIN, min(w, CROP_MAX) + 1, 4)
        if crop_h > h or crop_w > w:
            continue
        left = random.randint(0, w - crop_w)
        top = random.randint(0, h - crop_h)

        crop = img.crop((left, top, left + crop_w, top + crop_h))
        target = crop.resize((TARGET_SIZE, TARGET_SIZE), Image.BICUBIC)
        input_img = target.resize((TARGET_SIZE // 2, TARGET_SIZE // 2), Image.BICUBIC)

        inputs.append(to_tensor(input_img))
        targets.append(to_tensor(target))

for img_path in tqdm(image_paths, desc="Processing Images"):
    try:
        img = Image.open(img_path).convert("RGB")
        process_image(img)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

for vid_path in tqdm(video_paths, desc="Processing Videos"):
    try:
        duration = get_video_duration(vid_path)
    except Exception:
        continue
    if duration <= 0:
        continue
    for _ in range(FRAMES_PER_VIDEO):
        t = random.uniform(0, duration)
        try:
            frame = get_video_frame(vid_path, t)
            process_image(frame)
        except Exception:
            continue

inputs = torch.stack(inputs)
targets = torch.stack(targets)
torch.save((inputs, targets), OUTPUT_FILE)

print(f"Saved {len(inputs)} samples (input {inputs.shape[2:]} -> target {targets.shape[2:]}) to {OUTPUT_FILE}")
