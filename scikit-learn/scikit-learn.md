
---

# 🧠 scikit-learn দিয়ে কী কী করা যায়

---

# 🔹 1. Data Preprocessing 🧹

### 📌 কাজ:

* Missing value handle করা
* Feature scaling
* Encoding

### 🔧 Tools:

* `StandardScaler`, `MinMaxScaler`
* `LabelEncoder`, `OneHotEncoder`
* `SimpleImputer`

👉 Example:

```python
from sklearn.preprocessing import StandardScaler
```

---

# 🔹 2. Feature Engineering ⚙️

### 📌 কাজ:

* Feature selection
* Feature extraction
* Dimensionality reduction

### 🔧 Tools:

* `PCA` (Principal Component Analysis)
* `SelectKBest`
* `PolynomialFeatures`

---

# 🔹 3. Classification 🤖

👉 Output: category (spam / not spam)

### 🔧 Algorithms:

* Logistic Regression
* Naive Bayes
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)

---

# 🔹 4. Regression 📈

👉 Output: continuous value (price, score)

### 🔧 Algorithms:

* Linear Regression
* Ridge / Lasso
* Decision Tree Regressor
* Random Forest Regressor

---

# 🔹 5. Clustering 🧩

👉 Unsupervised learning

### 🔧 Algorithms:

* K-Means
* DBSCAN
* Agglomerative Clustering

---

# 🔹 6. Dimensionality Reduction 📉

👉 Feature কমানো

### 🔧 Tools:

* PCA
* TruncatedSVD

---

# 🔹 7. Model Evaluation 📊

### 📌 Metrics:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

```python
from sklearn.metrics import accuracy_score
```

---

# 🔹 8. Model Selection & Tuning 🎯

👉 Best model খুঁজে বের করা

### 🔧 Tools:

* `train_test_split`
* `cross_val_score`
* `GridSearchCV`
* `RandomizedSearchCV`

---

# 🔹 9. Pipeline 🚀

👉 সব step একসাথে automate করা

```python
from sklearn.pipeline import Pipeline
```

---

# 🔹 10. Text Processing (NLP Basics) 📝

👉 Spam detection-এর জন্য important

### 🔧 Tools:

* `CountVectorizer` (Bag of Words)
* `TfidfVectorizer`

---

# 🔹 11. Ensemble Methods 🌳

👉 Multiple model combine করা

### 🔧 Algorithms:

* Random Forest
* Gradient Boosting
* AdaBoost

---

# 🔹 12. Anomaly Detection 🚨

👉 Outlier detect করা

### 🔧 Tools:

* Isolation Forest
* One-Class SVM

---

# 🔹 13. Semi-Supervised Learning 🧠

👉 Partial labeled data দিয়ে train

---

# 🔹 14. Dataset Utilities 📂

### 🔧 Built-in datasets:

* Iris dataset
* Digits dataset
* Wine dataset

```python
from sklearn.datasets import load_iris
```

---

# 🔹 15. Model Saving 💾

👉 Model save/load করা

```python
import joblib
joblib.dump(model, "model.pkl")
```

---

# 🔥 Real Life তুমি কী কী বানাতে পারো?

✅ Spam Email Detector
✅ House Price Predictor
✅ Customer Segmentation
✅ Disease Prediction
✅ Recommendation System (basic)

---

# 🧩 Full Workflow (sklearn)

1. Load data
2. Preprocess
3. Feature extraction
4. Train model
5. Evaluate
6. Tune
7. Save

---

# 🧠 Short Summary

👉 scikit-learn = Complete ML toolkit
👉 Beginner → Intermediate সব কাজ করা যায়
👉 Deep learning ছাড়া প্রায় সব ML কাজ সম্ভব

---


