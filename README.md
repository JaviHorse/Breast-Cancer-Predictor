# Breast Cancer Tumor Classification Web App

A machine learning web app that classifies breast tumor profiles as **benign** or **malignant** using the **Wisconsin Diagnostic Breast Cancer dataset**.  

This project covers the full machine learning workflow: data preprocessing, model training, model evaluation, model comparison, model saving, and deployment through an interactive **Gradio** app hosted on **Hugging Face Spaces**.

> **Disclaimer:** This project is for educational and portfolio purposes only. It is not clinically validated and should not be used as a real medical diagnostic tool.

---

## Live Demo

Hugging Face Spaces:  
`https://huggingface.co/spaces/JaviHorse/breast-cancer-detection-app`

---

## Project Overview

This project uses machine learning to classify breast tumor profiles based on numerical diagnostic features extracted from digitized images of fine needle aspirates of breast masses.

The model predicts whether a tumor profile is:

- **Benign** — lower-risk / non-cancerous tumor profile
- **Malignant** — higher-risk / cancerous tumor profile

The final model was deployed as an interactive web app where users can input tumor measurement values and receive a real-time prediction.

---

## Dataset

The project uses the **Wisconsin Diagnostic Breast Cancer dataset**.

| Item | Description |
|---|---|
| Dataset | Wisconsin Diagnostic Breast Cancer Dataset |
| Samples | 569 |
| Input Features | 30 numerical tumor measurements |
| Target | Diagnosis |
| Classes | Benign, Malignant |
| Task | Binary Classification |

The dataset contains features related to tumor cell nuclei, including:

- radius
- texture
- perimeter
- area
- smoothness
- compactness
- concavity
- concave points
- symmetry
- fractal dimension

Each feature is provided in three forms:

- mean value
- standard error
- worst value

---

## Machine Learning Workflow

The project follows an end-to-end supervised machine learning pipeline:

1. Loaded the Wisconsin Diagnostic Breast Cancer dataset
2. Assigned proper column names to the raw dataset
3. Inspected the dataset shape, class distribution, and missing values
4. Encoded the diagnosis labels:
   - `B` → `0` for benign
   - `M` → `1` for malignant
5. Dropped the ID column since it was not useful for prediction
6. Split the data into training and testing sets
7. Applied feature scaling using `StandardScaler`
8. Trained and evaluated multiple machine learning models
9. Compared models using accuracy, precision, recall, F1-score, and confusion matrix
10. Selected the best-performing model
11. Saved the trained model, scaler, and feature names using `joblib`
12. Built and deployed an interactive Gradio web app

---

## Models Trained

The following machine learning models were trained and compared:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Support Vector Machine

---

## Model Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 96.49% | 97.50% | 92.86% | 95.12% |
| K-Nearest Neighbors | 95.61% | 97.44% | 90.48% | 93.83% |
| Decision Tree | 92.98% | 90.48% | 90.48% | 90.48% |
| Random Forest | 97.37% | 100.00% | 92.86% | 96.30% |
| Support Vector Machine | 97.37% | 100.00% | 92.86% | 96.30% |

The best-performing models were **Support Vector Machine** and **Random Forest**, both achieving:

- **97.37% test accuracy**
- **100.00% precision**
- **92.86% recall**
- **96.30% F1-score**

The final deployed model uses **Support Vector Machine**.

---

## Why Precision, Recall, and F1-Score Matter

Accuracy alone is not enough for medical-related classification problems.

For this project:

- **Precision** shows how many tumors predicted as malignant were actually malignant.
- **Recall** shows how many actual malignant tumors were correctly detected.
- **F1-score** balances precision and recall.
- **Confusion matrix** shows correct predictions, false positives, and false negatives.

In healthcare-related tasks, **false negatives are especially important** because they represent malignant tumors that were incorrectly classified as benign.

---

## Web App Features

The deployed Gradio app includes:

- Dark-mode responsive interface
- Input fields for all 30 diagnostic features
- Feature grouping into:
  - Mean Features
  - Standard Error Features
  - Worst Features
- Real-time prediction output
- Interpretation section explaining benign vs. malignant results
- Educational medical disclaimer
- Responsive layout for desktop and smaller screens

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn |
| Model Evaluation | accuracy, precision, recall, F1-score, confusion matrix |
| Model Saving | joblib |
| Web App | Gradio |
| Deployment | Hugging Face Spaces |

---

## Project Files

```text
breast-cancer-detection-app/
│
├── app.py                          # Gradio web application
├── requirements.txt                # Required Python packages
├── breast_cancer_svm_model.pkl     # Saved trained SVM model


## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/JaviHorse/Breast-Cancer-Predictor.git
cd Breast-Cancer-Predictor
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

For Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

For macOS/Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the app

```bash
python app.py
```

The app will run locally at:

```text
http://127.0.0.1:7860
```

---

## Requirements

```txt
gradio
scikit-learn
pandas
numpy
joblib
```

---

## How the App Works

The deployed app follows this process:

```text
User enters tumor feature values
        ↓
Input values are converted into a pandas DataFrame
        ↓
The saved StandardScaler scales the input
        ↓
The trained SVM model predicts the tumor class
        ↓
The app displays either Benign or Malignant
```
├── scaler.pkl                      # Saved StandardScaler
├── feature_names.pkl               # Saved feature names
└── README.md                       # Project documentation
