# SignalScope — Synthetic Media Detector

![SignalScope UI](https://truth-lens-spotter.lovable.app/og-image.png)

**SignalScope** is a media authenticity intelligence tool built for the **Smart India Hackathon (SIH) 2026**. It allows users to upload an image and receives a calibrated likelihood assessment of whether the image is **authentic (real)** or **synthetic (AI-generated)**. 

The tool prioritizes held-out performance on unseen AI generators and presents the results in a responsible, calibrated dashboard—communicating uncertainty rather than making binary accusations.

## 🚀 Features
* **AI vs Real Classification:** Powered by a PyTorch **ResNet18** binary classifier.
* **Calibrated Dashboard:** Displays overall model confidence and distinct probabilities for "Real" and "AI-generated".
* **Dynamic Advanced Explainer (UI):** 
  * Generates "Cues Behind the Verdict" dynamically (e.g., *Missing sensor noise*, *Spectral fingerprints*).
  * Generates a pseudo-realistic *Spectral Fingerprint chart*.
  * Displays a *Robustness to Degradation* table that simulates AUC drops across different compressions and retouches.
* **Lightning Fast:** Served via a lightweight FastAPI backend.

## 🛠️ Tech Stack
* **Backend:** Python, FastAPI, Uvicorn
* **Machine Learning:** PyTorch, Torchvision (ResNet18)
* **Frontend:** Vanilla HTML5, CSS3, JavaScript (Zero dependencies)

## ⚙️ Running Locally

### Prerequisites
1. Python 3.8+
2. A trained PyTorch model file (`best_model.pt`) configured for 2-class output (real/ai_generated).

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Nandan-00018/SIH_Falcon_Internal_SignalScope.git
   cd SIH_Falcon_Internal_SignalScope
   ```
2. Install the required Python dependencies:
   ```bash
   pip install fastapi uvicorn python-multipart torch torchvision pillow
   ```
3. Place your trained model file (`best_model.pth`) inside the `models/` directory at the root of the project.

### Starting the Server
Run the FastAPI backend server using Uvicorn:
```bash
python backend.py
```
*(Alternatively: `uvicorn backend:app --reload`)*

The frontend will be instantly accessible at: **http://127.0.0.1:8000**

## 🧩 Submission Structure
Per the hackathon guidelines (Section 4.1 & 7.3), this repository is structured as follows:

* `/model` — Contains the training/inference code and the required `predict` interface to evaluate the model.
* `/report` — Contains the one-page model report, detailing architecture, held-out performance, and failure analysis (plus explanation samples for Module A).
* `requirements.txt` — Environment file containing all necessary dependencies to run the project.
* `backend.py` — The FastAPI application serving the dashboard and handling the `/predict` inference endpoint.
* `static/index.html` — The complete frontend user interface (HTML/CSS/JS).

## 🏆 Hackathon Context
Developed for **SIH 2026 (Internal Hackathon)** under the domain of AI / Media Forensics / Trust & Safety.
