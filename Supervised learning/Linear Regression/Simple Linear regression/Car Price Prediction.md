

---

# ✅ **Project 2: Car Price Prediction (Using Horsepower)**

**Goal**: Predict the price of a car using its **horsepower** (single feature).

**Dataset**: [Car Dataset on Kaggle](https://www.kaggle.com/datasets/CooperUnion/cardataset)

---

## 🔧 Step-by-Step Implementation

---

### ✅ Step 1: Install Required Libraries (if not already installed)

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

### ✅ Step 2: Load the Dataset

```python
import pandas as pd

# Load CSV file (make sure you've downloaded the dataset)
df = pd.read_csv("data.csv")  # Adjust path if needed

# View basic info
print(df.head())
```

---

### ✅ Step 3: Select and Clean Relevant Columns

We’ll use:

* **`horsepower`** (X)
* **`price`** (Y)

```python
# Remove rows where horsepower or price is missing or '?'
df = df[df['horsepower'] != '?']
df = df.dropna(subset=['price'])

# Convert horsepower to float
df['horsepower'] = df['horsepower'].astype(float)
df['price'] = df['price'].astype(float)

# Select features
X = df[['horsepower']]
y = df['price']
```

---

### ✅ Step 4: Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```

---

### ✅ Step 5: Train Linear Regression Model

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

---

### ✅ Step 6: Make Predictions & Evaluate

```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)

print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
```

---

### ✅ Step 7: Plot the Results

```python
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='blue', label='Actual Price')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Horsepower')
plt.ylabel('Car Price ($)')
plt.title('Car Price Prediction')
plt.legend()
plt.grid(True)
plt.show()
```

---

### 🏁 Output

Example output:

```
Slope: 145.23
Intercept: 5000
R² Score: 0.68
```

✅ Interpretation:

* For every 1 unit increase in horsepower, price increases by about \$145.
* The regression line helps predict unseen car prices based on horsepower.

---



