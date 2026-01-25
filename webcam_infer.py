import argparse
import time

import cv2
import torch

from dataset import VideoDataset
from model import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live pain / no pain inference from webcam.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/pain_classifier_best.pth",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args.model_path, device=device)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    print("Press 'q' to quit.")

    frames_buffer = []
    last_infer_time = time.time()

    # We reuse VideoDataset's transforms by creating a tiny helper instance.
    helper_ds = VideoDataset(pain_dir="", no_pain_dir="", num_frames=args.num_frames, clips=[("", 0)])
    transform = helper_ds.transform

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_buffer.append(frame)
        if len(frames_buffer) > args.num_frames:
            frames_buffer.pop(0)

        display_text = "Collecting frames..."

        now = time.time()
        if len(frames_buffer) == args.num_frames and (now - last_infer_time) >= args.interval:
            last_infer_time = now
            # Prepare clip tensor
            processed = [transform(f[:, :, ::-1]) for f in frames_buffer]  # BGR->RGB
            clip_tensor = torch.stack(processed, dim=0).unsqueeze(0).to(device)  # (1, T, C, H, W)

            with torch.no_grad():
                logits = model(clip_tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0)
                prob_no_pain = probs[0].item()
                prob_pain = probs[1].item()
                label = "PAIN" if prob_pain >= prob_no_pain else "NO PAIN"
                conf = max(prob_pain, prob_no_pain)
                display_text = f"{label} ({conf:.2f})"

        # Overlay prediction text
        cv2.putText(
            frame,
            display_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255) if "PAIN" in display_text else (0, 255, 0),
            2,
        )
        cv2.imshow("Pain detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

