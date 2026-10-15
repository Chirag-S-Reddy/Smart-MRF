# Smart MRF: AI-Powered Material Recovery and Digital Twin

Hackathon project for **Problem Statement 06: Smart Recycling Assistant**.

Smart MRF detects waste items in images from a Material Recovery Facility (MRF) using a YOLO11 model, sorts each detection into a recycling bin, and shows the result in a web-based digital twin. Hazardous items (batteries and aerosols) raise an alert, and generative AI (Google Gemini) writes a plain-language report with recommendations for plant operators.

## How it works

```
image upload --> Flask backend (app.py) --> YOLO11 detection --> class-to-bin mapping --> web dashboard
                                                                        |
                                                                        +--> annotated image, counts per class and bin, hazard alert
                                                                        |
                                              twin snapshot --> /report --> Gemini (or offline summary) --> operator report
```

1. **Computer vision:** a YOLO11n model trained on a merged 8-class waste dataset detects items in each uploaded image.
2. **Backend:** `app.py` is a Flask server. It loads the model once at startup and exposes `/predict` (run detection) and `/health` (model and class info).
3. **Digital twin dashboard:** `templates/index.html` lets you upload an image, set the confidence threshold, and see the annotated image, per-class counts, bin totals and a hazard alert.
4. **Generative AI report:** the dashboard sends a snapshot of the twin (counts per bin and class, fill levels, hazard count, recent detections) to `POST /report`. `genai/report.py` turns it into a prompt for Gemini and returns a short report with a summary, hazards, bin status and recommendations. Without an API key it returns a clearly labelled rule-based summary instead, so the button always works.
5. **Command line detection:** `yolo_detect.py` runs the model on an image, a folder, a video file or a webcam.

## Detected classes (8)

`paper`, `cardboard`, `plastic`, `plastic_bag`, `glass`, `metal`, `battery`, `aerosol`

Each class maps to a twin bin: `plastic` and `plastic_bag` go to **plastic**, `paper` and `cardboard` go to **paper**, `glass` and `metal` keep their own bins, and `battery` and `aerosol` go to **hazardous**, which triggers the alert.

## Project status

| Part | Status |
|---|---|
| Dataset merge (3 sources to 8 classes) | Done |
| YOLO11n training (Colab, 40 epochs) | Done |
| Flask backend and web dashboard | Done |
| Generative AI report (`genai/report.py`, `/report`) | Done |
| Command line detection (`yolo_detect.py`) | Done |
| Digital twin state tracking (`twin/state.py`) | Done |
| Video tracking with persistent IDs | Done |

## Results

**1. Initial Model
Validation metrics after 40 epochs, from `train/results.csv`:

| Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|
| 0.700 | 0.594 | 0.659 | 0.448 |

The best mAP@50 (0.660) was at epoch 33. Training curves, the confusion matrix and sample predictions are in the `train/` folder. These numbers come from the validation split of the merged dataset, not from real belt footage.

**2. Final Version

| Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|

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

**Model weights**

The trained weights are included in the repository root: `best.pt` (best checkpoint, used by default) and `last.pt` (final-epoch checkpoint). Both sit next to `app.py`, so no extra download is needed. If you download only part of the repo and `best.pt` is missing, `app.py` falls back to `last.pt`, or you can point to any weights file with the `MODEL_PATH` setting below. You can also train your own model using the notebooks in `notebooks/` or the [Colab notebook](https://colab.research.google.com/drive/11z9iRRCZEEj_Qri5rhNmfURZ3SfXYd21?usp=sharing).

## Setup

Requirements: Python 3.9 or newer.

```bash
# 1. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows PowerShell: .\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
```

**Add your Gemini API key (for AI reports in case the provided one doesn't work)**

1. Create a free key at https://aistudio.google.com/apikey
2. Copy the example file and paste your key into it: 

```bash
cp .env.example .env              # Windows PowerShell: Copy-Item .env.example .env
```

```
GEMINI_API_KEY=your_key_here
```

That is all that is needed. The key stays on the server, is never sent to the browser, and `.env` is gitignored so it will not be committed. You can skip this step: the app still runs and the report button then gives an offline summary instead of an AI-written one.

Make sure `best.pt` (or `last.pt`) is in the project root, then start the app:

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser, scan an image of waste items, then use **Generate report** in the AI report panel. The panel states whether the report was written by Gemini or is the offline summary.

### Configuration

Both settings are optional environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `best.pt`, then `last.pt` next to `app.py` | Path to the weights file |
| `CONF_THRESHOLD` | `0.4` | Default minimum detection confidence |
| `GEMINI_API_KEY` | none | Enables Gemini-written reports (set it in `.env`) |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Gemini model used for reports |

Settings can go in `.env` or be set in the shell. Real environment variables win over `.env`. Example (Linux and macOS): `MODEL_PATH=/path/to/best.pt python app.py`

### API

- `GET /health` returns the loaded model name, class list, default confidence and the report mode (`gemini` or `offline`, never the key).
- `POST /predict` takes multipart form data with an image under the key `image` (JPG, PNG, BMP or WEBP, up to 16 MB) and an optional `conf` value between 0.05 and 0.95. It returns the detections, counts per class and per bin, a `hazardous` flag, the inference time in milliseconds and the annotated image.

- `POST /report` takes a JSON snapshot of the twin (`scans`, `bins`, `classes`, `avg_confidence`, `recent`) and returns `report` (Markdown), `source` (`gemini` or `offline`), `model` and an optional `warning`. If the Gemini call fails (bad key, quota, no internet), it falls back to the offline summary and explains why in `warning`.

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
├── .env.example          # copy to .env and add GEMINI_API_KEY
├── best.pt               # trained YOLO11n weights (best checkpoint)
├── last.pt               # trained YOLO11n weights (last epoch)
├── app.py                # Flask backend (/, /health, /predict)
├── yolo_detect.py        # command line detection on images, video or camera
├── templates/
│   └── index.html        # web dashboard
├── static/uploads/       # latest annotated image written by the backend
├── genai/
│   └── report.py         # Gemini report framework + offline fallback
├── tests/
│   └── test_report.py    # unit tests for the report module (mocked, no key needed)
├── twin/
│   └── state.py          # digital twin state (placeholder)
├── notebooks/            # Colab training notebook (.ipynb, .py and PDF exports)
└── train/                # training results: curves, confusion matrix, args.yaml, results.csv
```

## Tests

The report module has unit tests that mock the Gemini API, so they need no key and no internet:

```bash
python -m unittest discover tests
```

## Training

Training was done in Google Colab on a free T4 GPU, not in this repo. The merged dataset was built by remapping the labels of three Roboflow Universe datasets into the 8 classes above, keeping box annotations only and dropping classes that do not map. The model was trained with:

```bash
yolo detect train data=combined/data.yaml model=yolo11n.pt epochs=40 imgsz=640 batch=32 patience=10
```

The full workflow is in `notebooks/CV_Training_Notebook.ipynb`, and the exact settings used are in `train/args.yaml`.

**Run it in Colab:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11z9iRRCZEEj_Qri5rhNmfURZ3SfXYd21?usp=sharing)

Direct link: https://colab.research.google.com/drive/11z9iRRCZEEj_Qri5rhNmfURZ3SfXYd21?usp=sharing

In Colab, choose **Runtime > Change runtime type > T4 GPU** before running the cells. The datasets are not committed. The resulting weights (`best.pt` and `last.pt`) are committed in the repository root.

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
- Google Gemini API, for the generated reports
- Flask, for the backend
- OpenCV, NumPy and PyTorch

## License

This project depends on Ultralytics YOLO, which is licensed under AGPL-3.0, so this repository is released under the **AGPL-3.0** license. See `LICENSE.txt`.
