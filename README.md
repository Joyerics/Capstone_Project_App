# Social Media Usage Prediction System (CIUS-Based)

A machine learning-powered web application that predicts **high vs. non-high social media usage** and provides **research-backed recommendations** based on user input.

---

## Project Overview

This project applies machine learning techniques to analyze behavioral and demographic factors influencing social media usage.

We transitioned from traditional models to a more advanced **Artificial Neural Network (ANN)** to better capture complex, non-linear relationships in the data.

---

## Key Features

* Predicts **High vs Not-High Social Media Usage**
* Uses a trained **Artificial Neural Network (MLPClassifier)**
* Handles both **categorical and numerical inputs**
* Data preprocessing pipeline included
* Generates **research-based recommendations**
* Clean, SaaS-style user interface

---

## Model Architecture

**Pipeline:**

Raw Data → OneHot Encoding → Scaling → ANN → Prediction

**Model Details:**

* Hidden Layers: `(64, 32)`
* Activation Function: `ReLU`
* Optimizer: `Adam`
* Early Stopping: Enabled
* Output: Binary Classification

---

## 🔬 Model Evolution

| Model                         | Approach              | Outcome            |
| ----------------------------- | --------------------- | ------------------ |
| Logistic Regression (3-Class) | High / Moderate / Low | ❌ Poor separation  |
| Logistic Regression (Binary)  | High vs Not-High      | ⚠️ 63% accuracy    |
| ANN (Final Model)             | Non-linear learning   | ✅ Best performance |

---

## Why ANN?

Social media usage is influenced by complex interactions between variables such as:

* Age
* Income
* Anxiety
* Lifestyle

Linear models fail to capture these relationships.
ANN enables **non-linear pattern learning**, resulting in improved predictive performance.

---

## Research-Based Recommendations

The system provides recommendations based on academic research in:

* Behavioral psychology
* Digital wellbeing
* Social media addiction studies

Each recommendation in the app is backed by **peer-reviewed literature**.

---

## Project Structure

```
├── model/
│   ├── trained_model.pkl
│   ├── encoder.pkl
│   └── scaler.pkl
│
├── app/
│   ├── app.py
│   ├── templates/
│   ├── static/
│
├── notebook/
│   └── model_training.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Application

```bash
python app.py
```

---

## Deployment

This project is deployed using **Render**.

Ensure:

* `PORT` is used in your app
* `requirements.txt` is up to date
* All model files are included


---

## Timeline

Project Duration:
**September 2025 – April 2026**

---

## References

Research sources used for recommendations include:

* Digital Wellbeing Studies
* Social Media Usage & Mental Health Papers
* Behavioral Intervention Research

(Full citations included in project documentation)

---

## Final Note

This project demonstrates the application of machine learning in solving real-world behavioral prediction problems while integrating **academic research into actionable insights**.

---
