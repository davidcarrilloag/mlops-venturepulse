# 07 – CI/CD: Automated Training & Deployment with GitHub Actions

This module uses a modern CI/CD pipeline that automates the training, packaging,
and testing of our application. The main workflow (`ci-cd.yml`) calls a reusable
training job, builds a self-contained Docker image with the model "baked in," and
pushes it to the **GitHub Container Registry (GHCR)**.

Render is then used to pull this pre-built, validated image and run it as a live
web service.

---

## 🔁 Workflow Concept

```
Git Push
    │
    ▼
┌──────────────────────────────────────────┐
│ CI/CD Pipeline (ci-cd.yml)               │
│                                          │
│  1. Calls train.yml → creates artifact   │
│  2. Lints & tests the code               │
│  3. Builds & tests Docker image          │
│  4. Pushes image to GHCR                 │
│                                          │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│ GitHub Container Registry (GHCR)         │
│                                          │
│  Stores the final, versioned image:      │
│  ghcr.io/davidcarrilloag/mlops-venturepulse:latest │
│                                          │
└──────────────────────────────────────────┘
    │
    ▼ (Manual Deploy on Render)
┌──────────────────────────────────────────┐
│ Render.com                               │
│                                          │
│  Pulls the image from GHCR and runs it   │
│  as a live web service.                  │
│                                          │
└──────────────────────────────────────────┘
```

---

## 🗂️ Key Files

| File | Role |
| :--- | :--- |
| `train.py` | Trains the model and saves a production-ready copy to `models/model/`. |
| `app.py` | FastAPI service that loads the local `models/model/` at startup. |
| `test_api.py` | Automated tests run against the live Docker container to ensure quality. |
| `.github/workflows/ci-cd.yml` | **The main orchestrator**: calls training, lints, builds, tests, and pushes the image. |
| `.github/workflows/train.yml` | A **reusable component** dedicated solely to running `train.py`. |
| `Dockerfile` | Packages the code and the trained model into one container. |

**Note on `requirements.txt`:** The one in the root directory is for your local
development environment. The one inside `07-cicd/` is specifically for the
`Dockerfile`, ensuring the container has only the production dependencies it needs.

---

## 🚀 First-Time Deployment Guide

### 1️⃣ Validate Your Code Locally

```bash
flake8 07-cicd
```
No output means you're good to go.

### 2️⃣ Commit and Push

```bash
git add .
git commit -m "feat: add 07-cicd pipeline"
git push origin main
```
This triggers the CI/CD pipeline automatically.

### 3️⃣ Wait for the Pipeline to Succeed

Go to **GitHub → Actions** and wait for **VenturePulse CI/CD Pipeline** to complete.
The first run builds the Docker image and pushes it to GHCR (private by default).

### 4️⃣ Make the Docker Image Public (One-Time)

1. On your GitHub repo, go to **Packages** (right sidebar).
2. Click on `mlops-venturepulse`.
3. Go to **Package settings**.
4. In "Danger Zone", change visibility to **Public**.

### 5️⃣ Create the Render Service

1. On Render Dashboard: **New → Web Service**.
2. Choose **"Deploy an existing image from a registry"**.
3. Image URL: `ghcr.io/davidcarrilloag/mlops-venturepulse`
4. Select **Free** instance type → **Create Web Service**.

### 6️⃣ Verify Your Deployment

```bash
curl https://<your-service-name>.onrender.com/health
```

Expected response:
```json
{"status": "ok", "model_loaded": true, "run_id": "..."}
```

---

## 🔄 How to Redeploy

1. Make your changes.
2. `git commit` and `git push` to `main`.
3. Wait for the **CI/CD Pipeline** to complete on GitHub.
4. On Render: **Manual Deploy → Deploy latest commit**.
