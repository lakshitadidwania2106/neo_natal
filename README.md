## Pain vs No-Pain Video Classifier

This project trains a deep learning model to classify videos into **pain** vs **no pain**, and then uses that model to classify:

- **Any video file you provide**
- **Live camera feed** (webcam) in real time

The training script is pre-configured to use your dataset:

- Pain videos: `/Users/lucky21/Downloads/pain`
- No-pain videos: `/Users/lucky21/Downloads/no pain`

It uses **PyTorch** with **ResNet-18** (ImageNet-pretrained) applied on multiple frames per video, and averages predictions over time for higher accuracy.

---

### 1. Setup

From the `neo_natal` folder:

```bash
cd /Users/lucky21/Desktop/neo_natal
python3 -m venv .venv
source .venv/bin/activate  # on macOS / Linux
pip install -r requirements.txt
```

If `torch` fails to install, check the official install command for your Python version and OS, then rerun `pip install` accordingly.

---

### 2. Train the model on your dataset

Run:

```bash
cd /Users/lucky21/Desktop/neo_natal
source .venv/bin/activate
python train.py
```

Key points:

- By default it:
  - Reads videos from `/Users/lucky21/Downloads/pain` and `/Users/lucky21/Downloads/no pain`
  - Samples **16 frames** per video
  - Uses an **80% train / 20% validation** split
  - Trains for **20 epochs**
- The best model (highest validation accuracy) is saved as:
  - `models/pain_classifier_best.pth`

You can customize training, for example:

```bash
python train.py --epochs 30 --batch_size 2 --num_frames 24 --val_ratio 0.25
```

---

### 3. Classify a single video

After training (or if you already have `models/pain_classifier_best.pth`), you can classify any video:

```bash
python predict_video.py --video_path /path/to/your/video.mp4
```

Optional flags:

- `--model_path` (default: `models/pain_classifier_best.pth`)
- `--num_frames` (default: `16`)

Output looks like:

```text
Prediction: pain (confidence=0.93)
```

---

### 4. Live camera / webcam classification

To run real-time pain / no-pain inference from your webcam:

```bash
python webcam_infer.py
```

Notes:

- Make sure `models/pain_classifier_best.pth` exists (train first).
- Press **`q`** to quit the window.
- You can change the camera / settings:

```bash
python webcam_infer.py --camera_index 0 --interval 2.0 --num_frames 16
```

Where:

- `--camera_index` is the OpenCV camera index (0 is default webcam).
- `--interval` is seconds between predictions.
- `--num_frames` is the number of frames per prediction window.

---

### 5. How it works (high level)

- `dataset.py`
  - Builds a list of all videos and labels (1 = pain, 0 = no pain).
  - Loads each video with OpenCV.
  - Uniformly samples `num_frames` frames from start to end.
  - Applies ImageNet-style transforms (resize to 224×224, normalize).

- `model.py`
  - Uses a standard **ResNet-18** backbone (pretrained on ImageNet).
  - Treats each frame as an image, gets logits per frame.
  - Averages logits over all frames to get a **video-level prediction**.

- `train.py`
  - Splits the full list of videos into train / validation.
  - Trains with cross-entropy loss and Adam optimizer.
  - Tracks validation accuracy and saves the best checkpoint.

- `predict_video.py`
  - Loads the trained model and a single video.
  - Outputs `pain` / `no_pain` + confidence.

- `webcam_infer.py`
  - Captures a rolling window of frames from the webcam.
  - Every few seconds, runs the model and overlays a label on the video.

---

### 6. Tweaking for better accuracy

- **Increase `num_frames`** if your hardware allows (e.g., 24 or 32).
- **Train for more epochs**, or decrease learning rate (`--lr`).
- If the dataset is imbalanced (many more no-pain videos), consider:
  - Using `--batch_size 2` or 4 and training longer.
  - Adding class weights in `nn.CrossEntropyLoss` (can be computed from counts).

If you tell me your GPU / CPU specs and roughly how long you can train, I can suggest more aggressive hyperparameters for even higher accuracy.

