
---

# 🌦️ Weather Prediction System (Machine Learning + FastAPI)

This project is a **Machine Learning based Weather Prediction System** that predicts whether it will rain or not based on different weather conditions.

The model is trained using the **WeatherAUS dataset** and deployed as a **REST API using FastAPI**.

---

# 🚀 Project Overview

This project demonstrates a complete **Machine Learning workflow**:

* Data preprocessing
* Feature encoding
* Model training
* Model evaluation
* Model serialization
* API deployment using FastAPI

The trained model predicts **RainToday (Rain / No Rain)** using multiple weather features.

---

# 🧠 Machine Learning Pipeline

The following steps were used to build the model:

1. Load dataset
2. Handle missing values
3. Encode categorical features using LabelEncoder
4. Feature selection
5. Train-test split
6. Train machine learning models
7. Evaluate performance
8. Save model and scaler

Algorithms tested:

* Logistic Regression
* Decision Tree
* Random Forest

Final model and scaler are saved using **Joblib**.

---

# 📂 Project Structure

```
Weather Prediction System
│
├── app.py                     # FastAPI deployment
├── weather_predict.py         # Model training script
├── weather_prediction_model.pkl
├── scaler.pkl
├── weatherAUS.csv
├── requirements.txt
├── packages.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Mahfuzar148/Machine-Learning-and-Deep-learning.git
```

Go to the project directory:

```bash
cd Machine-Learning-and-Deep-learning
```

Create virtual environment:

```bash
python -m venv myvenv
```

Activate environment (Windows):

```bash
myvenv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the API

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Server will start at:

```
http://127.0.0.1:8000
```

---

# 📑 API Documentation

FastAPI automatically generates interactive documentation.

Swagger UI:

```
http://127.0.0.1:8000/docs
```

Here you can test the prediction endpoint.

---

# 📥 API Input Format

Example request body:

```json
{
  "data":[
    1,2,12.9,25.7,0,5,8,3,46,4,5,
    20,26,38,30,1007,1008,2,3,21,23,0
  ]
}
```

The API expects **22 feature values** in the same order used during model training.

---

# 📤 API Response

Example response:

```json
{
 "prediction": 0
}
```

Meaning:

```
0 → No Rain
1 → Rain
```

---

# 🛠 Technologies Used

* Python
* FastAPI
* Scikit-learn
* NumPy
* Pandas
* Joblib
* Uvicorn

---

# 📊 Dataset

Dataset used in this project:

```
WeatherAUS Dataset
```

Source:

Kaggle Weather Dataset

---

# 👨‍💻 Author

**Mahfuzar**

Machine Learning Enthusiast
Python Developer

GitHub:

```
https://github.com/Mahfuzar148
```

---

# ⭐ Future Improvements

Possible improvements for this project:

* Build a Streamlit Web App
* Deploy API to cloud (AWS / Render)
* Docker containerization
* Real-time weather data integration
* Frontend dashboard for predictions

---

If you find this project helpful, feel free to ⭐ the repository.

---

