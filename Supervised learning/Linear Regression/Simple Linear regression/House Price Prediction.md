
---

# ✅ **Project 1: House Price Prediction (Single Feature)**

**Dataset**: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

We’ll simplify it to use **one feature only**: `GrLivArea` (above ground living area in square feet) to predict `SalePrice`.

---

### 🔧 Step-by-Step Implementation

#### ✅ Step 1: Install Required Libraries

```bash
pip install pandas numpy matplotlib scikit-learn
```

#### ✅ Step 2: Load and Prepare Dataset

```python
import pandas as pd

# Load dataset (download manually and update the path)
data = pd.read_csv('train.csv')  # from House Prices Kaggle dataset

# Use only one feature for simplicity
data = data[['GrLivArea', 'SalePrice']]
data = data.dropna()  # Remove rows with missing values

X = data[['GrLivArea']]  # Independent variable
y = data['SalePrice']    # Dependent variable
```

---

#### ✅ Step 3: Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

#### ✅ Step 4: Train Linear Regression Model

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

---

#### ✅ Step 5: Evaluate the Model

```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)

print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
```

---

#### ✅ Step 6: Visualize the Regression Line

```python
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='blue', label='Actual Price')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Living Area (sq ft)')
plt.ylabel('Sale Price ($)')
plt.title('House Price Prediction')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 🏁 Final Output

* You’ll see a red regression line fitting house price vs living area.
* Output like:

  ```
  Slope: 100.8
  Intercept: 20000
  R² Score: 0.74
  ```

---

Would you like **Project 2** next? Here's a preview of the next few options:

| Project # | Title                                   | Dataset                                                                                                          |
| --------- | --------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 2         | Car Price Prediction (using Horsepower) | [Car Data](https://www.kaggle.com/datasets/CooperUnion/cardataset)                                               |
| 3         | Advertising Budget vs Sales             | [Advertising Dataset](https://www.kaggle.com/datasets/ishikajohari/advertising-dataset)                          |
| 4         | Salary Prediction Based on Experience   | [Simple Salary Dataset](https://www.kaggle.com/datasets/karthickveerakumar/salary-data-simple-linear-regression) |


