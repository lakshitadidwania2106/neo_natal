import argparse
from typing import Tuple

import torch

from dataset import VideoDataset
from model import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify a single video as pain / no pain.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/pain_classifier_best.pth",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_clip_tensor(video_path: str, num_frames: int) -> torch.Tensor:
    # Reuse VideoDataset's loader logic for a single clip.
    dummy_ds = VideoDataset(pain_dir="", no_pain_dir="", num_frames=num_frames, clips=[(video_path, 0)])
    clip_tensor, _ = dummy_ds[0]  # (T, C, H, W)
    return clip_tensor


def classify_video(
    video_path: str,
    model_path: str,
    num_frames: int,
    device_str: str,
) -> Tuple[str, float]:
    device = torch.device(device_str)
    model = load_model(model_path, device=device)

    clip = load_clip_tensor(video_path, num_frames=num_frames).unsqueeze(0).to(device)  # (1, T, C, H, W)
    with torch.no_grad():
        logits = model(clip)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    prob_no_pain = probs[0].item()
    prob_pain = probs[1].item()
    label = "pain" if prob_pain >= prob_no_pain else "no_pain"
    confidence = max(prob_pain, prob_no_pain)
    return label, confidence


def main() -> None:
    args = parse_args()
    label, conf = classify_video(
        video_path=args.video_path,
        model_path=args.model_path,
        num_frames=args.num_frames,
        device_str=args.device,
    )
    print(f"Prediction: {label} (confidence={conf:.4f})")


if __name__ == "__main__":
    main()

