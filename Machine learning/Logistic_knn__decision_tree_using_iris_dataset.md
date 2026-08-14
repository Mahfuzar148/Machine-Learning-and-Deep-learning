

1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Decision Tree**

প্রথমে common preprocessing, তারপর প্রতিটি algorithm-এর complete code।

---

# 1. Iris Dataset সম্পর্কে

Iris dataset-এ:

* **150 samples**
* **4 features**
* **3 classes**

### Features

```text
1. Sepal Length
2. Sepal Width
3. Petal Length
4. Petal Width
```

### Target

```text
0 → Setosa
1 → Versicolor
2 → Virginica
```

অর্থাৎ:

```text
Features (X) ──────► Model ──────► Target (y)
                                      │
                          Setosa / Versicolor / Virginica
```

---

# 2. Complete Supervised Learning Workflow

Exam-এ এই sequence মনে রাখবে:

```text
1. Import Libraries
       ↓
2. Load Dataset
       ↓
3. Explore Dataset
       ↓
4. Check Missing Values
       ↓
5. Separate X and y
       ↓
6. Train-Test Split
       ↓
7. Feature Scaling (যেখানে প্রয়োজন)
       ↓
8. Create Model
       ↓
9. Train Model
       ↓
10. Prediction
       ↓
11. Evaluation
       ↓
12. Confusion Matrix / Report
```

---

# 3. Step 1 — Import Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
```

---

# 4. Step 2 — Load Iris Dataset

```python
iris = load_iris()
```

এখন dataset-এর information দেখতে পারি:

```python
print(iris)
```

কিন্তু এত বড় output দরকার নেই।

তাই:

```python
print(iris.data)
print(iris.target)
```

---

# 5. Step 3 — Create X and y

```python
X = iris.data
y = iris.target
```

এখানে:

```text
X = Input Features
y = Target
```

Check:

```python
print("X shape:", X.shape)
print("y shape:", y.shape)
```

Output হবে approximately:

```text
X shape: (150, 4)
y shape: (150,)
```

অর্থাৎ:

```text
150 samples
4 features
```

---

# 6. Step 4 — Feature Names দেখো

```python
print("Feature Names:")
print(iris.feature_names)
```

Output:

```text
[
 'sepal length (cm)',
 'sepal width (cm)',
 'petal length (cm)',
 'petal width (cm)'
]
```

Target names:

```python
print("Target Names:")
print(iris.target_names)
```

Output:

```text
['setosa' 'versicolor' 'virginica']
```

---

# 7. Step 5 — DataFrame বানানো

এটা dataset বুঝতে খুব useful।

```python
df = pd.DataFrame(
    X,
    columns=iris.feature_names
)

df["target"] = y

print(df.head())
```

তখন dataset দেখতে হবে এরকম:

```text
sepal length    sepal width    petal length    petal width    target
5.1             3.5            1.4             0.2            0
4.9             3.0            1.4             0.2            0
6.2             3.4            5.4             2.3            2
```

---

# 8. Step 6 — Dataset Information

### Shape

```python
print(df.shape)
```

### Information

```python
print(df.info())
```

### Statistical information

```python
print(df.describe())
```

### Missing values

```python
print(df.isnull().sum())
```

Iris dataset-এ সাধারণত missing value নেই।

---

# 9. Step 7 — Separate Features and Target

যদিও আমরা আগে `X` এবং `y` করেছি, DataFrame থেকেও করা যায়:

```python
X = df.drop("target", axis=1)

y = df["target"]
```

এখন:

```text
X
↓
4 Features

y
↓
Target/Class
```

---

# 10. Step 8 — Train-Test Split

এটা supervised learning-এর খুব important step।

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

এখানে:

```text
test_size=0.2
```

মানে:

```text
80% → Training
20% → Testing
```

Iris-এর 150 sample হলে:

```text
120 → Training
30  → Testing
```

`random_state=42` দিলে একই split বারবার পাওয়া যায়।

`stratify=y` দিলে তিনটি class-এর proportion train/test-এ মোটামুটি বজায় থাকে।

---

# 11. Step 9 — Feature Scaling

সব algorithm-এর জন্য scaling প্রয়োজন হয় না।

### KNN → Scaling প্রয়োজন ✅

### Logistic Regression → Scaling করা ভালো ✅

### Decision Tree → Scaling প্রয়োজন নেই ❌

তাই KNN এবং Logistic Regression-এর জন্য:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

⚠️ খুব গুরুত্বপূর্ণ:

```python
fit_transform()
```

শুধু training data-তে।

Test data-তে:

```python
transform()
```

করবে।

---

# 12. Algorithm 1 — Logistic Regression

## Model তৈরি

```python
model_lr = LogisticRegression(
    max_iter=200
)
```

## Training

```python
model_lr.fit(
    X_train_scaled,
    y_train
)
```

## Prediction

```python
y_pred_lr = model_lr.predict(
    X_test_scaled
)
```

## Accuracy

```python
accuracy_lr = accuracy_score(
    y_test,
    y_pred_lr
)

print("Logistic Regression Accuracy:",
      accuracy_lr)
```

---

## Confusion Matrix

```python
cm_lr = confusion_matrix(
    y_test,
    y_pred_lr
)

print("Confusion Matrix:")
print(cm_lr)
```

---

## Classification Report

```python
print(
    classification_report(
        y_test,
        y_pred_lr,
        target_names=iris.target_names
    )
)
```

---

# 13. Algorithm 2 — KNN

KNN-এর ক্ষেত্রে আমরা scaled data ব্যবহার করব।

## Model

```python
model_knn = KNeighborsClassifier(
    n_neighbors=5
)
```

এখানে:

```text
K = 5
```

অর্থাৎ নতুন sample-এর classification করার সময় তার কাছের 5টি neighbor বিবেচনা করবে।

---

## Training

```python
model_knn.fit(
    X_train_scaled,
    y_train
)
```

---

## Prediction

```python
y_pred_knn = model_knn.predict(
    X_test_scaled
)
```

---

## Accuracy

```python
accuracy_knn = accuracy_score(
    y_test,
    y_pred_knn
)

print("KNN Accuracy:",
      accuracy_knn)
```

---

## Confusion Matrix

```python
print(
    confusion_matrix(
        y_test,
        y_pred_knn
    )
)
```

---

## Classification Report

```python
print(
    classification_report(
        y_test,
        y_pred_knn,
        target_names=iris.target_names
    )
)
```

---

# 14. Algorithm 3 — Decision Tree

Decision Tree-এর জন্য scaling প্রয়োজন নেই।

তাই এখানে:

```text
X_train
X_test
```

ব্যবহার করব।

## Model

```python
model_dt = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)
```

---

## Training

```python
model_dt.fit(
    X_train,
    y_train
)
```

---

## Prediction

```python
y_pred_dt = model_dt.predict(
    X_test
)
```

---

## Accuracy

```python
accuracy_dt = accuracy_score(
    y_test,
    y_pred_dt
)

print("Decision Tree Accuracy:",
      accuracy_dt)
```

---

## Confusion Matrix

```python
print(
    confusion_matrix(
        y_test,
        y_pred_dt
    )
)
```

---

## Classification Report

```python
print(
    classification_report(
        y_test,
        y_pred_dt,
        target_names=iris.target_names
    )
)
```

---

# 15. Compare All Algorithms

এখন:

```python
print("\n===== MODEL COMPARISON =====")

print(
    "Logistic Regression:",
    accuracy_lr
)

print(
    "KNN:",
    accuracy_knn
)

print(
    "Decision Tree:",
    accuracy_dt
)
```

এতে তিনটা model-এর accuracy compare করতে পারবে।

---

# 16. Complete Code — একসাথে

Exam-এর আগে এই versionটা **নিজে কয়েকবার run করবে**।

```python
# =====================================================
# SUPERVISED LEARNING USING IRIS DATASET
# =====================================================

# -----------------------------------------------------
# 1. Import Libraries
# -----------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# -----------------------------------------------------
# 2. Load Dataset
# -----------------------------------------------------

iris = load_iris()


# -----------------------------------------------------
# 3. Separate Features and Target
# -----------------------------------------------------

X = iris.data
y = iris.target


# -----------------------------------------------------
# 4. Display Dataset Information
# -----------------------------------------------------

print("Feature Names:")
print(iris.feature_names)

print("\nTarget Names:")
print(iris.target_names)

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)


# -----------------------------------------------------
# 5. Create DataFrame
# -----------------------------------------------------

df = pd.DataFrame(
    X,
    columns=iris.feature_names
)

df["target"] = y

print("\nFirst 5 Rows:")
print(df.head())


# -----------------------------------------------------
# 6. Dataset Information
# -----------------------------------------------------

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())


# -----------------------------------------------------
# 7. Train-Test Split
# -----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# -----------------------------------------------------
# 8. Feature Scaling
# -----------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# =====================================================
# MODEL 1: LOGISTIC REGRESSION
# =====================================================

print("\n\n===== LOGISTIC REGRESSION =====")

model_lr = LogisticRegression(
    max_iter=200
)

model_lr.fit(
    X_train_scaled,
    y_train
)

y_pred_lr = model_lr.predict(
    X_test_scaled
)

accuracy_lr = accuracy_score(
    y_test,
    y_pred_lr
)

print("Accuracy:", accuracy_lr)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred_lr
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_lr,
        target_names=iris.target_names
    )
)


# =====================================================
# MODEL 2: KNN
# =====================================================

print("\n\n===== KNN =====")

model_knn = KNeighborsClassifier(
    n_neighbors=5
)

model_knn.fit(
    X_train_scaled,
    y_train
)

y_pred_knn = model_knn.predict(
    X_test_scaled
)

accuracy_knn = accuracy_score(
    y_test,
    y_pred_knn
)

print("Accuracy:", accuracy_knn)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred_knn
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_knn,
        target_names=iris.target_names
    )
)


# =====================================================
# MODEL 3: DECISION TREE
# =====================================================

print("\n\n===== DECISION TREE =====")

model_dt = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

model_dt.fit(
    X_train,
    y_train
)

y_pred_dt = model_dt.predict(
    X_test
)

accuracy_dt = accuracy_score(
    y_test,
    y_pred_dt
)

print("Accuracy:", accuracy_dt)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred_dt
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_dt,
        target_names=iris.target_names
    )
)


# =====================================================
# MODEL COMPARISON
# =====================================================

print("\n\n===== MODEL COMPARISON =====")

print(
    "Logistic Regression Accuracy:",
    accuracy_lr
)

print(
    "KNN Accuracy:",
    accuracy_knn
)

print(
    "Decision Tree Accuracy:",
    accuracy_dt
)
```

---

# 17. Exam-এ Sir যদি শুধু "Iris Dataset দিয়ে supervised learning করো" বলেন

তাহলে তোমার মাথায় এই structure আসবে:

```text
Iris
 ↓
Classification
 ↓
X = features
y = target
 ↓
Train-Test Split
 ↓
Scaling
 ↓
Choose Algorithm
 ↓
Fit
 ↓
Predict
 ↓
Accuracy
 ↓
Confusion Matrix
 ↓
Classification Report
```

### সবচেয়ে সহজ algorithm হিসেবে

**Decision Tree** নিতে পারো:

```python
model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
```

আর যদি **KNN** নাও:

```text
Train-Test Split
        ↓
StandardScaler
        ↓
KNeighborsClassifier
        ↓
fit()
        ↓
predict()
        ↓
accuracy
```

---

## ⭐ Lab-এর জন্য এই ৮টা line সবচেয়ে বেশি মুখস্থ রাখো

```python
data = load_iris()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
```

**এই skeleton-এর logic বুঝে ফেললে**, Sir Iris-এর বদলে অন্য classification dataset দিলেও একই workflow ব্যবহার করতে পারবে—শুধু dataset loading এবং preprocessing অংশ পরিবর্তন হবে।
