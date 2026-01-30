# 🛡️ AI-Powered Multi-Modal Content Moderation System

> A full‑stack, production‑grade AI system for **automated moderation of images, videos, and text** using state‑of‑the‑art deep learning models, built with **React + Flask + PyTorch**.

---

## 🚀 About The Project

This project is a **production-grade, fully automated AI-powered content moderation system** built for a **large-scale social media platform**, where **every user-uploaded image, video, and text post is automatically analyzed in real time** before being published or escalated for human review.

The platform was designed to operate as a **high-throughput moderation pipeline** running on **AWS cloud infrastructure**, enabling:

* **Real-time ingestion of user-generated content (UGC)**
* **Automated multi-stage AI inference**
* **Dynamic risk scoring and content flagging**
* **Human-in-the-loop moderation workflows**
* **Centralized monitoring & analytics dashboard**

All uploaded content flows through a **distributed model inference stack hosted on AWS**, where multiple deep learning models operate in parallel to detect **violence, NSFW content, and hate speech**. Flagged content, risk metadata, and **user-generated abuse reports** are aggregated and surfaced in a **React-based moderation dashboard**, enabling moderators to review, approve, reject, or escalate cases.

The system is architected to simulate **real-world social media trust & safety pipelines**, capable of processing **thousands of concurrent uploads**, while maintaining **low-latency inference and high detection accuracy**.

> ⚠️ **Important:** The training datasets and pretrained model weights are **not included in this repository due to proprietary, privacy, and licensing constraints**. Only the full system architecture, inference pipelines, and deployment-ready application code are provided.

---

This project is a **comprehensive content moderation platform** designed to automatically detect and classify **harmful or unsafe content** across multiple modalities:

* 🖼️ **Images** – NSFW detection, body-part detection, nudity classification
* 🎥 **Videos** – Violence detection + frame‑based NSFW analysis
* 📝 **Text** – Hate speech detection + fine‑grained sub‑categorization

It is built to **simulate real-world social media moderation pipelines**, enabling:

* Automated AI‑based screening
* Human-in-the-loop moderation workflows
* Risk scoring & content flagging
* Review dashboards for moderation teams

> ⚠️ **Important:** The training datasets and pretrained model weights are **not included in this repository due to proprietary and licensing restrictions**. Only the full system architecture, inference pipelines, and deployment-ready code are provided.

---

## 🧠 System Architecture Overview

This platform follows a **multi-tier microservice-style architecture**:

```
React Dashboard  →  API Service Layer  →  Flask REST API  →  AI Inference Engine  →  Deep Learning Models
```

### High-Level Flow

1. User uploads **image/video + caption** from the dashboard.
2. React frontend sends request to backend API.
3. Flask server runs **multi-stage AI inference pipeline**.
4. AI models analyze content across **violence, NSFW, and hate speech** dimensions.
5. Aggregated results + confidence scores returned to frontend.
6. Dashboard shows **risk levels, flags, and moderation actions**.

---

## 🧩 Core AI Detection Pipeline

### 1️⃣ Violence Detection

* **Model:** VideoMAE (Transformer-based video classifier)
* **Input:** Video frames / images
* **Output:** Violence / Non-violence + confidence score

### 2️⃣ NSFW Detection (Two-Level System)

#### Level 1 — Fast Screening

* Transformer-based image classifier
* Categories: Porn / Sexy / Neutral / Drawings / Hentai

#### Level 2 — Precision Detection

* **YOLOv8-based object detector**
* Detects exposed body parts and explicit regions

> A bidirectional override logic ensures **high recall + high precision**.

### 3️⃣ Hate Speech Detection

* **Primary classifier:**

  * acceptable
  * inappropriate
  * offensive
  * violent

* **Subcategory classifier (conditional):**

  * racial_ethnic
  * religious
  * nationality
  * community
  * other

---

## 🖥️ Frontend Dashboard Features

* Drag & drop uploads
* Real-time moderation results
* Visual confidence indicators
* Risk-based tagging
* Human review controls
* Moderation queue management

> Screenshots will be added soon.

---

## ⚙️ Technology Stack

### Frontend

* React 18
* Tailwind CSS
* Axios
* Lucide Icons

### Backend

* Flask + Flask-CORS
* PyTorch
* HuggingFace Transformers
* OpenCV
* Ultralytics YOLOv8
* NumPy

### AI Models

* VideoMAE (Violence Detection)
* NSFW Transformer Classifier
* YOLOv8 NudeNet Detector
* BERT-based Hate Speech Model
* Fine‑tuned Subcategory Classifier

---

## 🗂️ Project Structure

```
backend/      → Flask API + AI inference pipeline
frontend/     → React moderation dashboard
test/         → Testing scripts & validation
```

---

## 🏗️ Deployment Architecture

### Development

```bash
Backend : python api_server.py
Frontend: npm start
```

### Production

* Backend → Gunicorn + Nginx
* Frontend → Static build → Netlify / S3
* Storage → AWS S3
* Database → PostgreSQL (moderation queue)

---


## 📜 License

This project is intended for **educational, research, and demonstration purposes**.

Commercial deployment requires **custom dataset licensing and compliance review**.

---

## 👨‍💻 Author

**Sudhanshu Swami**
AI & Backend Developer

---

> ⭐ If you find this project helpful, feel free to star the repository!
