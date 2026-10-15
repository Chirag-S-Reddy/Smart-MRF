"""Smart MRF Digital Twin - Flask backend.

Run:  python app.py      ->  http://127.0.0.1:5000

Endpoints: GET /, GET /health, POST /predict (image), POST /report (twin snapshot JSON),
POST /shutdown (stops the server; used by the Exit button).
"""
import base64
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO

from genai.report import generate_report, load_env, report_status

load_env()  # reads GEMINI_API_KEY etc. from .env (existing environment variables win)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CONF = float(os.getenv("CONF_THRESHOLD", "0.4"))

# Model class -> twin bin. Keys must match the class names in your training data.
CLASS_TO_BIN = {
    "plastic": "plastic",
    "plastic_bag": "plastic",
    "paper": "paper",
    "cardboard": "paper",
    "glass": "glass",
    "metal": "metal",
    "battery": "hazardous",
    "aerosol": "hazardous",
}
HAZARDOUS_BIN = "hazardous"


def find_model_path() -> Path:
    """MODEL_PATH env var wins; otherwise look for best.pt, then my_model.pt."""
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return Path(env_path)
    for name in ("best.pt", "last.pt"):
        candidate = BASE_DIR / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No model weights found. Put best.pt (or my_model.pt) next to app.py, "
        "or set MODEL_PATH."
    )


# Load once at startup (GPU is picked automatically if CUDA torch is installed).
MODEL_PATH = find_model_path()
model = YOLO(str(MODEL_PATH))
infer_lock = threading.Lock()  # one inference at a time, safe for threaded Flask
print(f"[smart-mrf] loaded {MODEL_PATH.name} with classes: {model.names}")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


def error(message: str, code: int):
    return jsonify({"status": "error", "message": message}), code


@app.errorhandler(413)
def too_large(_):
    return error("Image is larger than 16 MB.", 413)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model": MODEL_PATH.name,
            "classes": list(model.names.values()),
            "default_conf": DEFAULT_CONF,
            "report": report_status(),
        }
    )


@app.post("/predict")
def predict():
    file = request.files.get("image") or request.files.get("file")
    if file is None or file.filename == "":
        return error("No image uploaded. Send multipart form data with key 'image'.", 400)

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return error(
            f"Unsupported file type '{ext}'. Use: {', '.join(sorted(ALLOWED_EXTENSIONS))}.", 400
        )

    try:
        conf = float(request.form.get("conf", DEFAULT_CONF))
    except ValueError:
        conf = DEFAULT_CONF
    conf = min(max(conf, 0.05), 0.95)
    live = request.form.get("live") == "1"  # live camera mode: boxes only, skip the annotated image

    # Decode in memory (no temp file needed).
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return error("Could not read the image. The file may be corrupted.", 400)

    t0 = time.perf_counter()
    with infer_lock:
        result = model.predict(img, conf=conf, verbose=False)[0]
        annotated = None if live else result.plot()  # BGR image with boxes, labels, confidences
    inference_ms = round((time.perf_counter() - t0) * 1000)

    detections = []
    class_breakdown: dict[str, int] = {}
    bin_breakdown: dict[str, int] = {}

    if result.boxes is not None and len(result.boxes):
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clses = result.boxes.cls.cpu().numpy().astype(int)
        for box, c, k in zip(xyxy, confs, clses):
            name = model.names[int(k)]
            bin_name = CLASS_TO_BIN.get(name, "plastic")
            detections.append(
                {
                    "class": name,
                    "bin": bin_name,
                    "confidence": round(float(c), 3),
                    "bbox": [int(v) for v in box],
                }
            )
            class_breakdown[name] = class_breakdown.get(name, 0) + 1
            bin_breakdown[bin_name] = bin_breakdown.get(bin_name, 0) + 1

    if live:  # fast path for the live camera view: just the boxes
        return jsonify(
            {
                "status": "success",
                "total_count": len(detections),
                "detections": detections,
                "hazardous": bin_breakdown.get(HAZARDOUS_BIN, 0) > 0,
                "inference_ms": inference_ms,
            }
        )

    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return error("Failed to encode the annotated image.", 500)
    jpg_bytes = buf.tobytes()
    (UPLOAD_DIR / "annotated_latest.jpg").write_bytes(jpg_bytes)

    return jsonify(
        {
            "status": "success",
            "total_count": len(detections),
            "class_breakdown": class_breakdown,
            "bin_breakdown": bin_breakdown,
            "detections": detections,
            "hazardous": bin_breakdown.get(HAZARDOUS_BIN, 0) > 0,
            "inference_ms": inference_ms,
            "image_url": "/static/uploads/annotated_latest.jpg",
            "image_base64": "data:image/jpeg;base64," + base64.b64encode(jpg_bytes).decode(),
        }
    )


@app.post("/report")
def report():
    """Generate an operator report from a digital-twin snapshot (JSON body)."""
    snapshot = request.get_json(silent=True)
    if not isinstance(snapshot, dict):
        return error("Send a JSON snapshot of the twin state.", 400)
    result = generate_report(snapshot)
    return jsonify({"status": "success", **result})


@app.post("/shutdown")
def shutdown():
    """Stop the whole server (called by the Exit button in the UI).

    Only accepted from this machine, and only with the custom header the page sends.
    A custom header makes other websites unable to trigger it (browsers block it cross-origin).
    """
    if request.remote_addr not in ("127.0.0.1", "::1"):
        return error("Shutdown is only allowed from the local machine.", 403)
    if request.headers.get("X-Requested-With") != "smart-mrf":
        return error("Missing shutdown header.", 400)
    print("[smart-mrf] shutdown requested from the UI - exiting.")
    # Exit shortly after the response is sent so the browser receives it first.
    threading.Timer(0.5, lambda: os._exit(0)).start()
    return jsonify({"status": "success", "message": "Server shutting down."})


if __name__ == "__main__":
    # debug=False so the model is not loaded twice by the reloader.
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
