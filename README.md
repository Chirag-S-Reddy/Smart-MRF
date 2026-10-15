# Smart MRF: AI-Powered Material Recovery and Digital Twin

Hackathon project for **Problem Statement 06: Smart Recycling Assistant**.

Smart MRF watches a conveyor belt at a Material Recovery Facility (MRF), detects and tracks waste items with computer vision, mirrors the belt in a live digital twin, and uses generative AI to write plain-language reports and recommendations for plant operators.

## How it works

```
belt video --> YOLO11 detection + ByteTrack --> events.jsonl --> digital twin --> Streamlit dashboard
                                                                       |
                                                                       +--> Gemini report (recommendations)
```

1. **Computer vision:** a YOLO11 model detects waste items and tracks each one with a persistent ID, so every item is counted once.
2. **Digital twin:** `twin/state.py` turns the event stream into counts per material, bin fill levels, throughput, hazard alerts and a contamination estimate.
3. **Generative AI:** `genai/report.py` sends the twin state to Gemini and gets back a short report with recommendations.
4. **Dashboard:** a Streamlit app shows live counters, bin bars and the event table, and has a button to generate the AI report.

## Detected classes (8)

`paper`, `cardboard`, `plastic`, `plastic_bag`, `glass`, `metal`, `battery`, `aerosol`

`battery` and `aerosol` are treated as hazards by the twin and trigger alerts.

## Project status

| Part | Status |
|---|---|
| Repo structure | Done |
| Dataset merge (3 sources to 8 classes) | Done |
| YOLO11n training (Colab) | In progress |
| Fake event generator and twin | Planned |
| Dashboard and Gemini report | Planned |
| Real detections and tracking on belt video | Planned |
| Deployed replay mode | Planned |

Model accuracy numbers will be added here after training and evaluation. No results are claimed until then.

## Repository layout

```
smart-mrf/
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example          # GEMINI_API_KEY=your_key_here (the real .env is gitignored)
├── data/                 # data.yaml and dataset notes (images and labels are not committed)
├── src/                  # train.py, track.py, fake_events.py
├── twin/                 # state.py, the digital twin
├── genai/                # report.py, the Gemini report
├── dashboard/            # app.py, the Streamlit dashboard
├── sample/               # events.jsonl, used by the replay mode
├── docs/                 # slides and written material
└── assets/               # screenshots and demo images
```

## Setup

```bash
git clone https://github.com/Chirag-S-Reddy/Smart-MRF.git
cd Smart-MRF
pip install -r requirements.txt
cp .env.example .env      # then add your Gemini API key
```

On Windows PowerShell, use `Copy-Item .env.example .env`.

Run the dashboard (once `dashboard/app.py` is implemented):

```bash
streamlit run dashboard/app.py
```

## Training

Training is done in Google Colab on a free T4 GPU, not in this repo. The merged dataset is built by remapping the labels of three Roboflow Universe datasets into the 8 classes above, keeping box annotations only and dropping classes that do not map. The model is trained with:

```bash
yolo detect train data=combined/data.yaml model=yolo11n.pt epochs=40 imgsz=640 batch=32 patience=10
```

Datasets and weights (`*.pt`) are not committed. Weights are kept on Google Drive.

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
- Google Gemini, for the generated reports
- Streamlit, for the dashboard
- OpenCV and pandas

## License

This project depends on Ultralytics YOLO, which is licensed under AGPL-3.0, so this repository is released under the **AGPL-3.0** license. See `LICENSE`.
