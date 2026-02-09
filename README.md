# 🎓 Student Performance Prediction – Machine Learning Project

This project predicts student performance using Machine Learning and provides a web interface for real-time predictions.

---

## 🔗 Live Application

```text
https://student-performance-ml-predictor-jathin-dseehpf8aef9d2dm.centralindia-01.azurewebsites.net/predictdata
```

---

## 🧠 What This Project Does

```text
- Collects student details
- Processes data using ML pipelines
- Predicts student performance
- Displays results through a web application
```

---

## 🏗️ Project Architecture

```text
ml-project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   │
│   ├── exception.py
│   └── logger.py
│
├── templates/
│   ├── index.html
│   └── home.html
│
└── artifacts/
    ├── model.pkl
    └── preprocessor.pkl
```

---

## 🔄 Project Workflow (Simple Explanation)

```text
Raw Data
   ↓
Data Ingestion
   ↓
Data Transformation
   ↓
Model Training
   ↓
Model Saved
   ↓
User Input (Web Form)
   ↓
Prediction Output
```

---

## ⚙️ How the System Works

```text
1. Read and prepare data
2. Transform features
3. Train ML model
4. Save model and preprocessor
5. Load model in Flask app
6. Accept user input
7. Display prediction
```

---

## 🌐 Flask Web Routes

```text
/            → Home page
/predictdata → Prediction page
```

---

## ▶️ Run the Project Locally

```bash
git clone https://github.com/jathinreddy3515/ml-project.git
cd ml-project
pip install -r requirements.txt
python app.py
```

```text
http://127.0.0.1:10000
http://127.0.0.1:10000/predictdata
```

---

## 🚀 Production Server

```bash
gunicorn app:application
```

---

## ⚙️ App Settings (Azure)

```text
SCM_DO_BUILD_DURING_DEPLOYMENT = true
PYTHON_VERSION = 3.10
WEBSITES_PORT = 8000
```

---

## ☁️ Deployment Workflow

```text
GitHub Repository
   ↓
GitHub Actions
   ↓
Azure App Service
   ↓
Public URL
```

---

## 📦 Model Artifacts

```text
model.pkl
preprocessor.pkl
```

---

## ✅ Key Highlights

```text
- End-to-end ML project
- Modular architecture
- Flask web application
- Azure cloud deployment
- Production-ready setup
```

---

## 📌 Use Cases

```text
- Student performance analysis
- Education analytics
- ML portfolio project
- Interview demonstration
```

---

## 👨‍💻 Author

```text
Jathin Reddy
GitHub: https://github.com/jathinreddy3515
```




