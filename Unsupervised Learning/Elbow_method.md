
---

# 📈 Elbow Method – পূর্ণাঙ্গ ডকুমেন্টেশন

---

## 🎯 লক্ষ্য

Elbow Method ব্যবহার করে K-Means Clustering-এ উপযুক্ত K (ক্লাস্টারের সংখ্যা) নির্ধারণ করা হয়। এই পদ্ধতিতে আমরা K-এর বিভিন্ন মানের জন্য WCSS (Within Cluster Sum of Squares) হিসাব করি এবং K বনাম WCSS একটি গ্রাফ আঁকি। যেখানে WCSS হঠাৎ করে কমার হার কমে যায় — সেই বাঁক বা "elbow" হলো optimal K।

---

## 📐 সূত্র: WCSS (Within Cluster Sum of Squares)

$$
WCSS(K) = \sum_{j=1}^{K} \sum_{x_i \in cluster_j} \|x_i - \bar{x}_j\|^2
$$

* $x_i$: একটি data point
* $\bar{x}_j$: cluster j-এর mean বা centroid
* $K$: cluster এর সংখ্যা

---

## 🧪 বাস্তব উদাহরণ (ডেটাসেট)

আমরা নিচের প্রোডাক্টগুলোর Quantity ও Price ব্যবহার করছি:

| Product  | Quantity | Price (K) |
| -------- | -------- | --------- |
| FaceWash | 3        | 7         |
| Cream    | 5        | 4         |
| Shoes    | 4        | 3         |
| Bags     | 4        | 8         |
| Jacket   | 6        | 3         |
| Shirt    | 3        | 8         |

---

## 🧮 ধাপে ধাপে গণনা: WCSS Calculation

### ✅ K = 1:

সব পয়েন্ট একটি ক্লাস্টারে। centroid = (4.17, 5.5)

**Distance² Calculation:**

* FaceWash (3,7): (3−4.17)² + (7−5.5)² = 1.37 + 2.25 = 3.61
* Cream (5,4): (5−4.17)² + (4−5.5)² = 0.69 + 2.25 = 2.94
* Shoes (4,3): (4−4.17)² + (3−5.5)² = 0.03 + 6.25 = 6.28
* Bags (4,8): (4−4.17)² + (8−5.5)² = 0.03 + 6.25 = 6.28
* Jacket (6,3): (6−4.17)² + (3−5.5)² = 3.34 + 6.25 = 9.61
* Shirt (3,8): (3−4.17)² + (8−5.5)² = 1.37 + 6.25 = 7.61

📌 Total WCSS = 3.61 + 2.94 + 6.28 + 6.28 + 9.61 + 7.61 = **36.33**

---

### ✅ K = 2:

Cluster 1: FaceWash, Bags, Shirt → centroid = (3.33, 7.67) Cluster 2: Cream, Shoes, Jacket → centroid = (5, 3.33)

**Cluster 1 Calculations:**

* FaceWash (3,7): (3−3.33)² + (7−7.67)² = 0.11 + 0.45 = 0.56
* Bags (4,8): (4−3.33)² + (8−7.67)² = 0.45 + 0.11 = 0.56
* Shirt (3,8): (3−3.33)² + (8−7.67)² = 0.11 + 0.11 = 0.22

**Cluster 2 Calculations:**

* Cream (5,4): (5−5)² + (4−3.33)² = 0 + 0.45 = 0.45
* Shoes (4,3): (4−5)² + (3−3.33)² = 1 + 0.11 = 1.11
* Jacket (6,3): (6−5)² + (3−3.33)² = 1 + 0.11 = 1.11

📌 Total WCSS = (0.56+0.56+0.22) + (0.45+1.11+1.11) = 1.33 + 2.67 = **4.00**

---

### ✅ K = 1:

সব পয়েন্ট একটি ক্লাস্টারে — centroid: (4.17, 5.5)

**Distance² Calculation:**

* FaceWash → 3.61
* Cream → 2.94
* Shoes → 6.28
* Bags → 6.28
* Jacket → 9.61
* Shirt → 7.61

📌 Total WCSS = **36.33**

---

### ✅ K = 2:

ক্লাস্টার সংখ্যা ২ → নতুন দুটি centroid তৈরি হয়। প্রতিটি পয়েন্ট তার কাছের centroid এ অ্যাসাইন হয়।

**Cluster 1 Centroid:** (3.33, 7.67)  → FaceWash, Bags, Shirt
**Cluster 2 Centroid:** (5, 3.33)     → Cream, Shoes, Jacket

**WCSS Calculation:**

* Cluster 1: 0.56 + 0.56 + 0.22 = 1.33
* Cluster 2: 0.45 + 1.11 + 1.11 = 2.67

📌 Total WCSS = **4.00**

---

### ✅ K = 3:

WCSS = **2.33**  → ছোট আরও ক্লাস্টার, পয়েন্টগুলো আরও কাছাকাছি

### ✅ K = 4:

WCSS = **1.33**

### ✅ K = 5:

WCSS = **0.50**

---

## 📊 WCSS মানের তালিকা

| K (Clusters) | Total WCSS |
| ------------ | ---------- |
| 1            | 36.33      |
| 2            | 4.00       |
| 3            | 2.33       |
| 4            | 1.33       |
| 5            | 0.50       |

---

## 📍 Elbow Point বিশ্লেষণ

K = 2 এ বিশাল পরিমাণে WCSS কমেছে (36.33 → 4.00)। K = 3 তে আরও কমেছে তবে পরিবর্তনের হার কম। K = 4 ও 5 তে হ্রাসের হার খুবই ধীর।

📌 তাই **Elbow Point = K = 3**, কারণ সেখান থেকে efficiency কমে গেছে।

---

## 📈 Python কোড (গ্রাফ এবং WCSS গণনা সহ ধাপে ধাপে ব্যাখ্যা)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from numpy.linalg import norm

# Sample dataset
X = pd.DataFrame({
    'Product': ['FaceWash', 'Cream', 'Shoes', 'Bags', 'Jacket', 'Shirt'],
    'Quantity': [3, 5, 4, 4, 6, 3],
    'Price': [7, 4, 3, 8, 3, 8]
})

features = X[['Quantity', 'Price']]

k_values = [1, 2, 3, 4, 5]
wcss = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(features)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    wcss_k = 0
    print(f"
K = {k} Clusters:")
    for i, point in enumerate(features.values):
        centroid = centroids[labels[i]]
        dist_sq = np.sum((point - centroid) ** 2)
        wcss_k += dist_sq
        print(f"  Point {X['Product'][i]} {point} → Centroid {centroid} → Distance² = {dist_sq:.4f}")
    print(f"Total WCSS for K={k}: {wcss_k:.2f}")
    wcss.append(round(wcss_k, 2))

# Plot the Elbow graph
plt.plot(k_values, wcss, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.axvline(x=3, color='red', linestyle='--', label='Elbow Point (K=3)')
plt.legend()
plt.grid(True)
plt.show()
```

এই কোডে প্রতিটি `K` মানের জন্য:

* প্রতিটি ডেটা পয়েন্ট কোন ক্লাস্টারে পড়েছে তা দেখা যায়,
* সেই ক্লাস্টারের **centroid** থেকে কতটা দূরত্ব (স্কোয়ার) তা দেখানো হয়,
* সবগুলো distance² যোগ করে **Total WCSS** হিসাব করা হয়।

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from numpy.linalg import norm
import numpy as np

# Sample dataset
X = pd.DataFrame({
    'Quantity': [3, 5, 4, 4, 6, 3],
    'Price': [7, 4, 3, 8, 3, 8]
})

k_values = [1, 2, 3, 4, 5]
wcss = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    wcss_k = 0
    for i, label in enumerate(labels):
        wcss_k += norm(X.iloc[i] - centroids[label]) ** 2
    wcss.append(round(wcss_k, 2))

# Plot the Elbow graph
plt.plot(k_values, wcss, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.axvline(x=3, color='red', linestyle='--', label='Elbow Point (K=3)')
plt.legend()
plt.grid(True)
plt.show()

# Display individual WCSS values
for i, val in enumerate(wcss):
    print(f"K={k_values[i]}, WCSS={val}")
```

```python
import matplotlib.pyplot as plt
k_values = [1, 2, 3, 4, 5]
wcss = [36.33, 4.00, 2.33, 1.33, 0.50]

plt.plot(k_values, wcss, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.axvline(x=3, color='red', linestyle='--', label='Elbow Point (K=3)')
plt.legend()
plt.grid(True)
plt.show()
```

---

## ✅ উপসংহার

* Elbow Method সহজ ও শক্তিশালী পদ্ধতি clustering optimization-এর জন্য
* WCSS হঠাৎ যেখানে কমার হার থেমে যায় — সেটাই Elbow Point
* আমাদের ডেটার ক্ষেত্রে Optimal K = **3**
* এটি ব্যবহার করে আমরা dataset-কে যথাযথভাবে ক্লাস্টার করতে পারি

---
