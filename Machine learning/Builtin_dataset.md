
---

# 🔥 1. Classification Dataset

এখানে **target/class label থাকে** এবং output category হয়।

| Dataset           | কী predict করবে  | Type           | Load                   |
| ----------------- | ---------------- | -------------- | ---------------------- |
| **Iris**          | Flower species   | Classification | `load_iris()`          |
| **Breast Cancer** | Malignant/Benign | Classification | `load_breast_cancer()` |
| **Wine**          | Wine class       | Classification | `load_wine()`          |
| **Digits**        | Digit 0–9        | Classification | `load_digits()`        |

---

## 1️⃣ Iris Dataset ⭐⭐⭐⭐⭐

সবচেয়ে important।

```python
from sklearn.datasets import load_iris

data = load_iris()

X = data.data
y = data.target

print(X.shape)
print(y.shape)
print(data.feature_names)
print(data.target_names)
```

### Type:

```text
Classification
```

### Features:

```text
sepal length
sepal width
petal length
petal width
```

### Target:

```text
Setosa
Versicolor
Virginica
```

### Practice:

* Logistic Regression
* KNN
* Decision Tree
* Random Forest
* SVM

---

# 2️⃣ Breast Cancer Dataset ⭐⭐⭐⭐⭐

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = data.data
y = data.target

print(X.shape)
print(y.shape)
print(data.feature_names)
print(data.target_names)
```

### Type:

```text
Classification
```

Target:

```text
Malignant
Benign
```

### Practice:

* Logistic Regression
* KNN
* Decision Tree
* SVM
* Random Forest

---

# 3️⃣ Wine Dataset ⭐⭐⭐⭐

```python
from sklearn.datasets import load_wine

data = load_wine()

X = data.data
y = data.target

print(X.shape)
print(y.shape)
print(data.feature_names)
print(data.target_names)
```

### Type:

```text
Classification
```

Target:

```text
3 different wine classes
```

Practice:

* Logistic Regression
* KNN
* Decision Tree
* Random Forest

---

# 4️⃣ Digits Dataset ⭐⭐⭐⭐

```python
from sklearn.datasets import load_digits

data = load_digits()

X = data.data
y = data.target

print(X.shape)
print(y.shape)
print(data.target_names)
```

### Type:

```text
Classification
```

এখানে handwritten digit:

```text
0 1 2 3 4 5 6 7 8 9
```

predict করতে হয়।

---

# 🔵 2. Regression Dataset

এখানে target সাধারণত **continuous numerical value**।

---

# 5️⃣ Diabetes Dataset ⭐⭐⭐⭐⭐

Linear Regression practice করার জন্য খুব useful।

```python
from sklearn.datasets import load_diabetes

data = load_diabetes()

X = data.data
y = data.target

print(X.shape)
print(y.shape)
print(data.feature_names)
```

### Type:

```text
Regression
```

### Target:

Continuous numerical value.

Practice:

* Linear Regression
* Ridge
* Lasso
* Random Forest Regressor

---

# 🟢 3. Unsupervised Dataset

এগুলোতে সাধারণত **target label থাকে না**।

Scikit-learn-এ unsupervised practice-এর জন্য synthetic dataset generator বেশি useful।

---

# 6️⃣ make_blobs ⭐⭐⭐⭐⭐

K-Means practice-এর জন্য খুব ভালো।

```python
from sklearn.datasets import make_blobs

X, y = make_blobs(
    n_samples=300,
    centers=3,
    n_features=2,
    random_state=42
)

print(X.shape)
print(y.shape)
```

এখানে `y` generator-এর তৈরি cluster information; **K-Means practice-এ model-কে `y` দেবে না**।

```python
from sklearn.cluster import KMeans

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

labels = model.fit_predict(X)

print(labels)
```

### Type:

```text
Unsupervised → Clustering
```

### Practice:

* K-Means
* Hierarchical Clustering
* DBSCAN

---

# 7️⃣ make_moons ⭐⭐⭐⭐

Non-linear clustering বুঝতে খুব ভালো।

```python
from sklearn.datasets import make_moons

X, y = make_moons(
    n_samples=300,
    noise=0.05,
    random_state=42
)

print(X.shape)
```

এখানেও clustering practice করার সময় `y` ব্যবহার করবে না।

---

# 8️⃣ make_circles ⭐⭐⭐⭐

Circular/non-linear cluster practice করার জন্য।

```python
from sklearn.datasets import make_circles

X, y = make_circles(
    n_samples=300,
    noise=0.05,
    factor=0.5,
    random_state=42
)

print(X.shape)
```

Practice:

* K-Means
* DBSCAN
* Hierarchical Clustering

---

# 📌 পুরো list একসাথে

### 🔴 Supervised → Classification

```text
1. Iris
2. Breast Cancer
3. Wine
4. Digits
```

Load করার functions:

```python
load_iris()
load_breast_cancer()
load_wine()
load_digits()
```

---

### 🔵 Supervised → Regression

```text
5. Diabetes
```

Load:

```python
load_diabetes()
```

---

### 🟢 Unsupervised → Clustering

```text
6. make_blobs
7. make_moons
8. make_circles
```

Load:

```python
make_blobs()
make_moons()
make_circles()
```

---

# ⭐ তোমার ML Lab-এর জন্য কোনগুলো সবচেয়ে বেশি practice করবে?

আমি হলে এই **৫টা** আগে শেষ করতাম:

```text
                 ML LAB
                   │
        ┌──────────┴──────────┐
        │                     │
   SUPERVISED            UNSUPERVISED
        │                     │
   ┌────┴────┐                │
   │         │                │
Classification Regression   Clustering
   │         │                │
   ▼         ▼                ▼
 Iris     Diabetes        make_blobs
   │
   ├── Logistic
   ├── KNN
   └── Decision Tree
```

### Priority:

**1. Iris → Logistic + KNN + Decision Tree** ⭐⭐⭐⭐⭐
**2. Diabetes → Linear Regression** ⭐⭐⭐⭐⭐
**3. Breast Cancer → Classification** ⭐⭐⭐⭐
**4. make_blobs → K-Means** ⭐⭐⭐⭐⭐
**5. make_moons → DBSCAN/K-Means** ⭐⭐⭐

---

# ⚠️ একটা গুরুত্বপূর্ণ বিষয়

`load_iris()` দিয়ে dataset load করলে সাধারণত:

```python
data = load_iris()

X = data.data
y = data.target
```

এই দুইটা সবচেয়ে important।

**`X` = input/features**

**`y` = output/target**

কিন্তু `make_blobs()`-এর ক্ষেত্রে:

```python
X, y = make_blobs(...)
```

K-Means করার সময়:

```python
model.fit_predict(X)
```

করবে—**`y` model-কে দেবে না**, কারণ K-Means হলো unsupervised।

---

## 🧠 Exam-এর জন্য এই mapping মুখস্থ রাখো

```text
Iris              → Classification
Breast Cancer     → Classification
Wine              → Classification
Digits            → Classification

Diabetes          → Regression

make_blobs        → Clustering
make_moons        → Clustering
make_circles      → Clustering
```

এগুলো জানলে built-in dataset নিয়ে practice করার জন্য যথেষ্ট ভালো foundation তৈরি হবে।
