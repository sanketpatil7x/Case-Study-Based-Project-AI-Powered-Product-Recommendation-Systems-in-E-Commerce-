# 🎯 AI-Powered Recommendation System

## 📌 Overview

This project implements an end-to-end **AI-based recommendation system** using collaborative filtering (Matrix Factorization).
It generates personalized item recommendations based on user interaction data.

---

## 🚀 Key Features

* End-to-end ML pipeline (data → model → evaluation → UI)
* Matrix Factorization using SVD
* Top-K recommendation generation
* Evaluation metrics: Precision@K, Recall@K, NDCG
* Interactive UI using Streamlit
* Hyperparameter tuning and model comparison

---

## 🧠 System Architecture

User Input → Model (SVD) → Ranking → Top-K Recommendations → UI Display

---

## 📊 Dataset

* MovieLens 100K dataset
* 943 users, 1682 items
* ~100K interactions

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd recommender-system

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
python -m streamlit run api/app.py
```

---

## 📈 Model Performance

| Metric       | Score |
| ------------ | ----- |
| Precision@10 | ~0.30 |
| Recall@10    | ~0.19 |
| NDCG@10      | ~0.36 |

---

## 🔬 Experiments

| Model      | Precision@10 |
| ---------- | ------------ |
| Popularity | ~0.10        |
| MF (k=20)  | ~0.30        |
| MF (k=50)  | ~0.25        |
| MF (k=100) | ~0.19        |

---

## 🧠 Key Insights

* Optimal latent dimension improves performance
* Model significantly outperforms popularity baseline (~3x)
* Data sparsity impacts recommendation quality
* Overfitting occurs at high latent dimensions

---

## ⚠️ Limitations

* Cold-start problem not handled
* No content-based features
* Offline evaluation only

---

## 🔮 Future Work

* Hybrid recommendation system
* Deep learning models (Two-Tower, Transformers)
* Real-time recommendations
* Deployment on cloud

---

## 📂 Project Structure

```
recommender-system/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│
├── src/
├── api/
│   └── app.py
│
├── requirements.txt
├── README.md
```

---

## 👨‍💻 Author

Sanket Patil

---
