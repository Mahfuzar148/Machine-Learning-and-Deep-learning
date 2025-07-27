
---

## 🔷 **Linear Regression (in Supervised Learning)**

### ✅ **Definition**

Linear Regression is a **regression algorithm** used to **predict a continuous dependent variable** based on one or more independent variables by fitting a linear relationship (a straight line) to the observed data.

---

## 🧠 **Core Idea**

> The goal is to model the relationship between input variables $X$ and output variable $y$ using a linear equation:

$$
y = wX + b
$$

* $y$ = predicted output
* $X$ = input features
* $w$ = weight (slope or coefficient)
* $b$ = bias (intercept)

---

## 📈 **Types of Linear Regression**

| Type                           | Description                                                                                           |
| ------------------------------ | ----------------------------------------------------------------------------------------------------- |
| **Simple Linear Regression**   | One input variable (e.g., house size → price).                                                        |
| **Multiple Linear Regression** | Multiple input variables (e.g., size, location, bedrooms → price).                                    |
| **Polynomial Regression**      | A non-linear version by including powers of input features, but still modeled linearly in parameters. |

---

## 📊 **Use Cases**

* Predicting house prices
* Stock market forecasting
* Sales prediction
* Temperature estimation
* Risk modeling

---

## 🧾 **Assumptions of Linear Regression**

1. **Linearity**: The relationship between X and y is linear.
2. **Independence**: Observations are independent of each other.
3. **Homoscedasticity**: Constant variance of residuals.
4. **No multicollinearity**: Input variables shouldn't be highly correlated.
5. **Normality of errors**: Residuals should be normally distributed.

---

## 🔢 **Mathematical Objective**

The model finds parameters $w$ and $b$ that **minimize the error** between predicted and actual values.

### 🎯 **Loss Function (Mean Squared Error):**

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Where:

* $y_i$ = actual value
* $\hat{y}_i$ = predicted value
* $n$ = number of observations

---

## 🛠️ **Evaluation Metrics**

* **Mean Absolute Error (MAE)**
* **Mean Squared Error (MSE)**
* **Root Mean Squared Error (RMSE)**
* **R-squared ($R^2$)**: Proportion of variance explained by the model.

---

## 🐍 **Python Implementation Example (with scikit-learn)**

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Sample data
data = pd.DataFrame({
    'Area': [1000, 1500, 2000, 2500, 3000],
    'Price': [200000, 300000, 400000, 500000, 600000]
})

# Features and label
X = data[['Area']]
y = data['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Coefficients
print("Slope (w):", model.coef_[0])
print("Intercept (b):", model.intercept_)
```

---

## 📌 **Advantages**

* Simple and fast
* Easy to interpret
* Useful as a baseline model
* Works well with linearly separable data

### ⚠️ **Limitations**

* Assumes linearity
* Sensitive to outliers
* Doesn’t perform well with complex relationships
* Requires careful feature selection and assumption validation

---


