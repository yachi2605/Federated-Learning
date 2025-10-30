# 🧠 Federated Learning and AI Model Serving

**Author:** Yachi Darji  
**Check report :** [Open PDF](./report/Report.pdf)
---


## 📘 Project Overview

This project implements:

1. **Federated Learning (FL) Simulation**  
   - Simulates multiple clients with **non-IID** data using **ThreadPoolExecutor** for concurrent local training.  
   - Trains a global model (`model.pth`) based on the **FedAvg algorithm**.  
   - Logs round-wise metrics and saves them to a CSV file.

2. **AI Model Serving Web Application**  
   - A **FastAPI** web app that loads the trained global model for inference.  
   - Users can upload an image and view **top-5 predictions** with labels on the same page.  
   - The app is **containerized with Docker** for easy deployment on **Chameleon Cloud**.

---

## 📂 Folder Structure

```
a/
├── fl_simulation/
│   ├── config.py
│   ├── data_utils.py
│   ├── federated_learning.py
│   ├── model_utils.py
│   ├── run.py
│
├── web_app/
│   └── app.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── main.py
```

---

## ⚙️ Environment Setup

### Option 1 — Run with Docker (Recommended)

#### Build and Run Containers
```bash
docker compose up --build
```

This will:
- Train the federated model (`fl-training` service)
- Launch the web application (`fl-webapp` service) on **port 8000**

Access the app at:
```
http://<YOUR_PUBLIC_IP>:8000
```

#### Stop Containers
```bash
docker compose down
```

---

### Option 2 — Run Without Docker

#### Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Run Federated Learning Simulation
```bash
python -m fl_simulation.run
```

Outputs:
- Trained model → `results/model.pth`
- Training logs → `logs/fl.log`
- Round metrics → `results/training_results.csv`

#### Run Web Application
```bash
uvicorn web_app.app:app --host 0.0.0.0 --port 8000
```

Then visit:
```
http://127.0.0.1:8000
```

---

## 🌐 Access on Chameleon Cloud

1. Launch your instance with a **Floating IP**.  
2. Ensure port **8000** is open in your **security group**.  
3. Upload your `a.zip`:
   ```bash
   scp -i "/path/to/key.pem" a.zip cc@<YOUR_PUBLIC_IP>:~/
   ssh -i "/path/to/key.pem" cc@<YOUR_PUBLIC_IP>
   unzip a.zip -d ~/a
   cd ~/a
   docker compose up --build
   ```
4. Open your browser and visit:
   ```
   http://<YOUR_PUBLIC_IP>:8000
   ```

---

## 🧪 Testing the Web App

1. Upload any image (e.g., a CIFAR-10 or random photo).  
2. Click **Predict**.  
3. The app will show **top-5 predicted class labels** with confidence scores.  
4. All predictions are logged in:
   ```
   logs/webapp.log
   ```

---

## 📊 Outputs and Artifacts

| File | Description |
|------|--------------|
| `results/training_results.csv` | Per-round metrics (train/test loss & accuracy) |
| `results/model.pth` | Final trained global model |
| `results/data_distribution.png` | Non-IID client data visualization |
| `logs/fl.log` | Federated training log file |
| `logs/webapp.log` | Inference request logs |
| `docker-compose.yml` | Defines training and web app containers |

---

## ✅ Verification Checklist for Grader

| Requirement | Check |
|--------------|-------|
| **Non-IID Data** | Verified by `results/data_distribution.png` |
| **Correct Executor Use** | `ThreadPoolExecutor` used in `federated_learning.py` |
| **FL Simulation** | `python -m fl_simulation.run` runs successfully |
| **Web App Works** | Accessible via `http://<PUBLIC_IP>:8000` |
| **Containerized** | Dockerfile & Compose setup included |
| **Report Submitted** | PDF report attached separately |

---

## 🧹 Clean Up After Testing

```bash
docker compose down
docker system prune -af
```

---

## 🧠 Notes

- Python version used: **3.10+**  
- Docker version: **25+**, Docker Compose v2  
- Works on both **local (Mac/Linux)** and **Chameleon Cloud** environments  
- No dataset included — CIFAR-10/100 is downloaded automatically by PyTorch.

---

## 🏁 End of README

Once running, you should see:
- Federated Learning rounds with accuracy improving each round.  
- A functional FastAPI UI showing top-5 image predictions on the same page.  

---

