import os
import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class VideoDataset(Dataset):
    """
    Simple video dataset for binary classification (pain / no pain).

    It samples a fixed number of frames from each video, applies image
    transforms, and returns a tensor of shape (T, C, H, W).
    """

    def __init__(
        self,
        pain_dir: str,
        no_pain_dir: str,
        num_frames: int = 16,
        transform: transforms.Compose | None = None,
        clips: List[Tuple[str, int]] | None = None,
    ) -> None:
        """
        If `clips` is provided it should be a list of (video_path, label)
        and pain_dir / no_pain_dir are ignored except for reproducibility.
        """
        super().__init__()
        self.num_frames = num_frames
        self.transform = transform or self._default_transform()

        if clips is not None:
            self.samples = clips
        else:
            self.samples = self._build_index(pain_dir, no_pain_dir)

    @staticmethod
    def _build_index(pain_dir: str, no_pain_dir: str) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        for root, _, files in os.walk(pain_dir):
            for f in files:
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    samples.append((os.path.join(root, f), 1))  # 1 = pain
        for root, _, files in os.walk(no_pain_dir):
            for f in files:
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    samples.append((os.path.join(root, f), 0))  # 0 = no pain
        return samples

    @staticmethod
    def _default_transform() -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]
        frames = self._load_and_sample_frames(video_path)
        # frames: list of HxWxC (BGR) numpy arrays
        processed = [self.transform(frame[:, :, ::-1]) for frame in frames]  # BGR->RGB
        clip_tensor = torch.stack(processed, dim=0)  # (T, C, H, W)
        return clip_tensor, label

    def _load_and_sample_frames(self, path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            raise RuntimeError(f"Video has no frames: {path}")

        # Choose indices uniformly across the video
        indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)

        frames: List[np.ndarray] = []
        current_idx = 0
        target_pos = indices[0]
        success, frame = cap.read()

        while success and len(frames) < self.num_frames:
            if current_idx == target_pos:
                frames.append(frame)
                if len(frames) == self.num_frames:
                    break
                target_pos = indices[len(frames)]
            success, frame = cap.read()
            current_idx += 1

        cap.release()

        # If we got fewer frames than requested, pad by repeating last
        if not frames:
            raise RuntimeError(f"Failed to read frames from: {path}")

        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        return frames


def train_val_split(
    pain_dir: str,
    no_pain_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """
    Simple random train/val split over video clips.
    """
    all_samples = VideoDataset._build_index(pain_dir, no_pain_dir)
    rng = random.Random(seed)
    rng.shuffle(all_samples)

    n_total = len(all_samples)
    n_val = int(n_total * val_ratio)
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]
    return train_samples, val_samples

