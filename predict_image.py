import argparse

import cv2
import torch

from dataset import VideoDataset
from model import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify a single image as pain / no pain.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/pain_classifier_best.pth",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of virtual frames to create from the image.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    return parser.parse_args()


def classify_image(
    image_path: str,
    model_path: str,
    num_frames: int,
    device_str: str,
) -> tuple[str, float]:
    device = torch.device(device_str)

    # Load model
    model = load_model(model_path, device=device)

    # Reuse dataset transforms
    helper_ds = VideoDataset(pain_dir="", no_pain_dir="", num_frames=num_frames, clips=[("", 0)])
    transform = helper_ds.transform

    # Read image (BGR)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise RuntimeError(f"Could not read image at: {image_path}")

    # BGR -> RGB, then transform
    img_rgb = img_bgr[:, :, ::-1]
    frame_tensor = transform(img_rgb)  # (C, H, W)

    # Create a "virtual video" by repeating the frame
    clip = torch.stack([frame_tensor] * num_frames, dim=0)  # (T, C, H, W)
    clip = clip.unsqueeze(0).to(device)  # (1, T, C, H, W)

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
    label, confidence = classify_image(
        image_path=args.image_path,
        model_path=args.model_path,
        num_frames=args.num_frames,
        device_str=args.device,
    )
    print(f"Prediction: {label} (confidence={confidence:.4f})")


if __name__ == "__main__":
    main()

