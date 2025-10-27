# app.py
import io
import base64
from typing import List, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms as T
from PIL import Image

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)

NUM_CLASSES = 100
WEIGHTS_PATH = "web_app/global_model_final.pth"

# CIFAR-100 label names 
CIFAR100_LABELS = [
    "apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle","bicycle","bottle",
    "bowl","boy","bridge","bus","butterfly","camel","can","castle","caterpillar","cattle",
    "chair","chimpanzee","clock","cloud","cockroach","couch","crab","crocodile","cup","dinosaur",
    "dolphin","elephant","flatfish","forest","fox","girl","hamster","house","kangaroo",
    "keyboard","lamp","lawn_mower","leopard","lion","lizard","lobster","man","maple_tree",
    "motorcycle","mountain","mouse","mushroom","oak_tree","orange","orchid","otter","palm_tree",
    "pear","pickup_truck","pine_tree","plain","plate","poppy","porcupine","possum","rabbit","raccoon",
    "ray","road","rocket","rose","sea","seal","shark","shrew","skunk","skyscraper","snail",
    "snake","spider","squirrel","streetcar","sunflower","sweet_pepper","table","tank","telephone","television",
    "tiger","tractor","train","trout","tulip","turtle","wardrobe","whale","willow_tree","wolf","woman","worm"
][:NUM_CLASSES]

# normalization 
preprocess = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
])

# -----------------------------
# Model (ResNet18 for CIFAR-like input)
# -----------------------------
def build_model(num_classes: int) -> nn.Module:
    m = models.resnet18(weights=None)
    # CIFAR-style stem (no 7x7/stride-2)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

model = build_model(NUM_CLASSES)
try:
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    # supports either raw state_dict or wrapped
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[app.py] Warning: missing keys: {missing}, unexpected keys: {unexpected}")
except FileNotFoundError:
    print(f"[app.py] WARNING: weights file not found at {WEIGHTS_PATH}. "
          f"The app will run with random weights (predictions will be nonsense).")

model.to(DEVICE).eval()

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Assignment 2 Image Classifier UI", version="1.0")

# -----------------------------
# Helpers
# -----------------------------
def read_image_to_tensor(upload: UploadFile) -> torch.Tensor:
    upload.file.seek(0)
    data = upload.file.read()
    if not data:
        raise ValueError(f"{upload.filename}: empty or unreadable file")
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return preprocess(img)

def pil_to_data_url(img: Image.Image) -> str:
    # generate an inline thumbnail (for display only)
    thumb = img.copy()
    thumb.thumbnail((160, 160))
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def topk_labels(probs_row: torch.Tensor, k: int = 5) -> List[str]:
    vals, idxs = torch.topk(probs_row, k)
    labels = []
    for i in idxs.tolist():
        if 0 <= i < len(CIFAR100_LABELS):
            labels.append(CIFAR100_LABELS[i])
        else:
            labels.append(str(i))
    return labels

# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    # Simple single-page UI
    html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Image Classifier</title>
<style>
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; background:#0b1020; color:#e7ecff;}
  h1 { margin: 0 0 8px; font-size: 24px;}
  p { margin: 0 0 16px; opacity: .85;}
  form { margin: 16px 0 28px; padding: 16px; border: 1px solid #334; border-radius: 12px; background: #121a33;}
  input[type=file] { margin-right: 8px; }
  button { padding: 10px 14px; border-radius: 10px; border: 0; background:#5562ff; color:white; cursor:pointer; }
  button:hover { filter: brightness(1.08); }
  .grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; }
  .card { background:#121a33; border:1px solid #334; border-radius: 14px; padding: 12px; }
  .thumb { width: 100%; border-radius: 10px; background:#0e1430; aspect-ratio: 1/1; object-fit: contain; }
  .name { margin-top: 10px; font-size: 14px; color:#aab3ff; }
  .labels { margin: 8px 0 0; padding-left: 18px; }
  .labels li { margin: 4px 0; }
  .footer { margin-top: 26px; font-size: 12px; opacity:.7;}
</style>
</head>
<body>
  <h1>Assignment 2 Image Classifier UI</h1>
  <p>Upload one or more images and click <b>Predict</b>.</p>

  <form id="uploadForm" action="/predict" method="post" enctype="multipart/form-data">
    <input type="file" name="files" accept="image/*" multiple required />
    <button type="submit">Predict</button>
  </form>

  <div id="results"></div>

  <script>
    const form = document.getElementById('uploadForm');
    const results = document.getElementById('results');

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      results.innerHTML = "<p>Running inference…</p>";

      const formData = new FormData(form);
      const res = await fetch('/api/predict', { method: 'POST', body: formData });
      if (!res.ok) {
        const err = await res.json().catch(()=>({detail:'Unknown error'}));
        results.innerHTML = "<p style='color:#ff8a8a'>Error: " + (err.detail || res.statusText) + "</p>";
        return;
      }
      const data = await res.json();
      const items = data.results || [];

      const grid = document.createElement('div');
      grid.className = 'grid';

      for (const item of items) {
        const card = document.createElement('div');
        card.className = 'card';
        const img = document.createElement('img');
        img.className = 'thumb';
        img.src = item.preview_data_url;
        img.alt = item.filename;

        const name = document.createElement('div');
        name.className = 'name';
        name.textContent = item.filename;

        const ul = document.createElement('ul');
        ul.className = 'labels';
        for (const label of item.top5_labels) {
          const li = document.createElement('li');
          li.textContent = label; // labels only (no percentages)
          ul.appendChild(li);
        }

        card.appendChild(img);
        card.appendChild(name);
        card.appendChild(ul);
        grid.appendChild(card);
      }

      results.innerHTML = "";
      results.appendChild(grid);
      results.insertAdjacentHTML('beforeend', '<div class="footer">Done.</div>');
    });
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html)

@app.post("/predict", response_class=HTMLResponse)
async def predict_page(files: List[UploadFile] = File(...)):
   
    return index()

@app.post("/api/predict")
async def api_predict(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    tensors = []
    thumbs = []
    names = []

    
    for f in files:
        try:
            f.file.seek(0)
            raw = f.file.read()
            if not raw:
                raise ValueError("empty file")
            pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
            tensor = preprocess(pil_img)
            tensors.append(tensor)
            thumbs.append(pil_to_data_url(pil_img))
            names.append(f.filename or "image")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read {getattr(f, 'filename', 'file')}: {e}")

    # Single batched forward pass; align outputs with inputs by index
    batch = torch.stack(tensors, dim=0).to(DEVICE)
    with torch.inference_mode():
        logits = model(batch)
        probs = F.softmax(logits, dim=1).cpu()

    results = []
    for name, thumb, p in zip(names, thumbs, probs):
        labels = topk_labels(p, k=5)  # labels only, per your request
        results.append({
            "filename": name,
            "top5_labels": labels,
            "preview_data_url": thumb,
        })

    return JSONResponse({"results": results})

# -----------------------------
# Dev server entrypoint
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    # Run:  uvicorn app:app --reload
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)