# Smart MRF: AI-Powered Material Recovery and Digital Twin

Hackathon project for **Problem Statement 06: Smart Recycling Assistant**.

Smart MRF detects waste items in images from a Material Recovery Facility (MRF) using a YOLO11 model, sorts each detection into a recycling bin, and shows the result in a web-based digital twin. Hazardous items (batteries and aerosols) raise an alert.

## How it works

```
image upload --> Flask backend (app.py) --> YOLO11 detection --> class-to-bin mapping --> web dashboard
                                                                        |
                                                                        +--> annotated image, counts per class and bin, hazard alert
```

1. **Computer vision:** a YOLO11n model trained on a merged 8-class waste dataset detects items in each uploaded image.
2. **Backend:** `app.py` is a Flask server. It loads the model once at startup and exposes `/predict` (run detection) and `/health` (model and class info).
3. **Digital twin dashboard:** `templates/index.html` lets you upload an image, set the confidence threshold, and see the annotated image, per-class counts, bin totals and a hazard alert.
4. **Command line detection:** `yolo_detect.py` runs the model on an image, a folder, a video file or a webcam.

## Detected classes (8)

`paper`, `cardboard`, `plastic`, `plastic_bag`, `glass`, `metal`, `battery`, `aerosol`

Each class maps to a twin bin: `plastic` and `plastic_bag` go to **plastic**, `paper` and `cardboard` go to **paper**, `glass` and `metal` keep their own bins, and `battery` and `aerosol` go to **hazardous**, which triggers the alert.

## Project status

| Part | Status |
|---|---|
| Dataset merge (3 sources to 8 classes) | Done |
| YOLO11n training (Colab, 40 epochs) | Done |
| Flask backend and web dashboard | Done |
| Command line detection (`yolo_detect.py`) | Done |
| Digital twin state tracking (`twin/state.py`) | Planned (file is a placeholder) |
| Video tracking with persistent IDs | Planned |
| Gemini-generated reports | Planned |

## Results

Validation metrics after 40 epochs, from `train/results.csv`:

| Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|
| 0.700 | 0.594 | 0.659 | 0.448 |

The best mAP@50 (0.660) was at epoch 33. Training curves, the confusion matrix and sample predictions are in the `train/` folder. These numbers come from the validation split of the merged dataset, not from real belt footage.

## Download

You can get the project in either of two ways.

**Option 1: Download the ZIP (no Git needed)**

1. Open https://github.com/Chirag-S-Reddy/Smart-MRF
2. Click the green **Code** button, then **Download ZIP**. You can also use the direct link: https://github.com/Chirag-S-Reddy/Smart-MRF/archive/refs/heads/master.zip
3. Extract the ZIP. You will get a folder named `Smart-MRF-master`.
4. Open a terminal inside that folder and continue with **Setup** below.

**Option 2: Clone with Git**

```bash
git clone https://github.com/Chirag-S-Reddy/Smart-MRF.git
cd Smart-MRF
```

**Download the model weights (required)**

The trained weights are not stored in this repository (`*.pt` files are gitignored). Download `best.pt` from the project's Google Drive folder, at `smart-mrf-runs/v1/weights/best.pt`, and place it in the project root next to `app.py`. If you do not have access, ask the team, or train your own model using the notebooks in `notebooks/`.

## Setup

Requirements: Python 3.9 or newer.

```bash
# 1. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows PowerShell: .\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
```

Make sure `best.pt` is in the project root, then start the app:

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser, upload an image of waste items, and click detect.

### Configuration

Both settings are optional environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `best.pt`, then `last.pt` next to `app.py` | Path to the weights file |
| `CONF_THRESHOLD` | `0.4` | Default minimum detection confidence |

Example (Linux and macOS): `MODEL_PATH=/path/to/best.pt python app.py`

### API

- `GET /health` returns the loaded model name, class list and default confidence.
- `POST /predict` takes multipart form data with an image under the key `image` (JPG, PNG, BMP or WEBP, up to 16 MB) and an optional `conf` value between 0.05 and 0.95. It returns the detections, counts per class and per bin, a `hazardous` flag, the inference time in milliseconds and the annotated image.

```bash
curl -F "image=@sample.jpg" -F "conf=0.4" http://127.0.0.1:5000/predict
```

### Command line detection

```bash
python yolo_detect.py --model best.pt --source test.jpg
python yolo_detect.py --model best.pt --source video.mp4 --thresh 0.4
python yolo_detect.py --model best.pt --source usb0 --resolution 1280x720
```

`--source` accepts an image, a folder, a video file, `usb0` (USB camera) or `picamera0`. While the window is open, press `q` to quit, `s` to pause and `p` to save a frame. This script opens a display window, so if it fails with an OpenCV GUI error, install the full build with `pip uninstall opencv-python-headless` and then `pip install opencv-python`.

## Repository layout

```
Smart-MRF/
├── README.md
├── LICENSE.txt
├── requirements.txt
├── app.py                # Flask backend (/, /health, /predict)
├── yolo_detect.py        # command line detection on images, video or camera
├── templates/
│   └── index.html        # web dashboard
├── static/uploads/       # latest annotated image written by the backend
├── twin/
│   └── state.py          # digital twin state (placeholder)
├── notebooks/            # Colab training notebook (.ipynb, .py and PDF exports)
└── train/                # training results: curves, confusion matrix, args.yaml, results.csv
```

## Training

Training was done in Google Colab on a free T4 GPU, not in this repo. The merged dataset was built by remapping the labels of three Roboflow Universe datasets into the 8 classes above, keeping box annotations only and dropping classes that do not map. The model was trained with:

```bash
yolo detect train data=combined/data.yaml model=yolo11n.pt epochs=40 imgsz=640 batch=32 patience=10
```

The full workflow is in `notebooks/CV_Training_Notebook.ipynb`, and the exact settings used are in `train/args.yaml`. Datasets and weights are not committed. Weights are kept on Google Drive.

## Credits and attributions

**Team**
- Chirag S Reddy (DSATM): project lead and developer
- Nikshay MS (BIT)
- Chetan B (BMS)
- Gurusmaran Raghavan (BIT)

**AI assistance**
- Claude, an AI assistant made by Anthropic, helped with project planning, repo setup, the dataset-merge and training workflow, and this README.

**Datasets** (Roboflow Universe, all licensed CC BY 4.0)
- Recyclable Waste, by thesis-ifxof: https://universe.roboflow.com/thesis-ifxof/recyclable-waste
- dataset, by jrabitu: https://universe.roboflow.com/jrabitu/dataset-7blb4
- YOLO Waste Detection, by Phatpharid's Workspace: https://universe.roboflow.com/phatpharids-workspace/yolo-waste-detection-tyahu

Labels from these datasets were remapped and merged into the 8 classes listed above. The original datasets belong to their authors.

**Tools and libraries**
- Ultralytics YOLO11 (AGPL-3.0): https://github.com/ultralytics/ultralytics
- Roboflow Universe, for hosting the datasets: https://universe.roboflow.com
- Google Colab, for training
- Flask, for the backend
- OpenCV, NumPy and PyTorch

## License

This project depends on Ultralytics YOLO, which is licensed under AGPL-3.0, so this repository is released under the **AGPL-3.0** license. See `LICENSE.txt`.
