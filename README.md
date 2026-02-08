## End to End Deployment of  Machine Learning Project

\# 📦 ML Project — End-to-End Machine Learning Application



This repository contains a complete end-to-end Machine Learning project built with \*\*Python\*\*, including:



✔ Data ingestion, preprocessing, and transformation  

✔ Model training and evaluation  

✔ Saving preprocessors and models as pickle files  

✔ A Flask web app to predict student math scores from user input



---



\## 🧠 Project Overview



This project takes student performance data, builds a regression model to predict math scores, and provides a \*\*web interface\*\* so users can enter features and get predictions in real time.



It follows a clean workflow with reusable modules, custom exceptions, and logging.



---



\## 🗂️ Project Structure



ml-project/

│

├── src/

│ ├── components/

│ │ ├── data\_ingestion.py

│ │ ├── data\_transformation.py

│ │ └── model\_trainer.py

│ ├── pipeline/

│ │ └── predict\_pipeline.py

│ ├── utils.py

│ ├── logger.py

│ └── exception.py

│

├── templates/

│ ├── index.html

│ └── home.html

│

├── artifacts/

│ ├── preprocessor.pkl

│ └── model.pkl

│

├── app.py

├── requirements.txt

└── README.md





---



\## 🚀 Setup Instructions



\### 1️⃣ Clone the repository

```bash

git clone https://github.com/jathinreddy3515/ml-project.git

cd ml-project



2️⃣ Create and activate a Python environment

python -m venv venv

venv\\Scripts\\activate      # Windows

\# OR

source venv/bin/activate   # macOS / Linux



3️⃣ Install dependencies

pip install -r requirements.txt





This installs:



Flask (for web app)



scikit-learn (ML tools)



pandas + numpy (data handling)



catboost, xgboost (models)



📊 How the Pipeline Works



Data Ingestion



Load original dataset



Split into train/test



Save CSVs



Data Transformation



Handle missing values



Encode categorical features



Scale numerical features



Save preprocessor.pkl



Model Training



Train multiple regression models



Evaluate using R² score



Save best model as model.pkl



Web App Prediction



User enters inputs in HTML form



Flask loads saved preprocessor + model



Predicts math score in real time



🏃 Running Locally

🔹 Train the model

python src/components/data\_ingestion.py



🔹 Start the web app

python app.py





Open your browser:



http://127.0.0.1:5000/





Submit the form to get predicted math scores.



📌 Usage Example



Enter values like:



Gender: Female



Race/Ethnicity: group B



Lunch Type: Standard



Reading Score: 70



Writing Score: 75



Click Predict, and the predicted math score will appear.



🗃️ Notes



Ensure preprocessor.pkl and model.pkl exist in artifacts/ after training.



Run data transformation and model training first if missing.





❓ Questions



Verify Python version (3.8+ recommended)



Ensure virtual environment is active



Dependencies installed correctly



📝 Summary



This project demonstrates complete flow from raw data to a usable ML-powered web app. It is a solid learning base for ML projects and can be extended with more models or deployment features.

