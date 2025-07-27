
---

## ✅ Step 1: Choose a Dataset from Kaggle

Let’s use a popular Kaggle dataset for this example:

🔗 **Dataset**: [Student Scores Dataset (Hours vs Scores)](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

This dataset has:

* `Hours`: Number of hours studied (independent variable X)
* `Scores`: Percentage score (dependent variable Y)

---

## ✅ Step 2: Code Implementation in Python (Using `pandas`, `matplotlib`, `sklearn`)

```python
# 📌 Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 📌 Load the dataset
url = "http://bit.ly/w-data"  # This is a direct link to the CSV file
data = pd.read_csv(url)

# 📌 View the first few rows
print(data.head())

# 📌 Define independent and dependent variables
X = data[['Hours']]  # must be 2D
y = data['Scores']   # 1D

# 📌 Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 📌 Make predictions
y_pred = model.predict(X_test)

# 📌 Print coefficients
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)

# 📌 Visualize regression line with training data
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.plot(X_train, model.predict(X_train), color='red', linewidth=2, label='Regression line')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.title('Hours vs Score - Training Set')
plt.legend()
plt.grid(True)
plt.show()

# 📌 Evaluate the model
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))
```

---

## ✅ Output You’ll Get

* The **regression line** will be plotted on top of the training data.
* You’ll see printed values:

  * **Slope** $m$
  * **Intercept** $c$
  * **MSE (Error)** and **R² (accuracy score)**

---

## ✅ Model Equation Example

After training, the model might output:

```
Slope (m): 9.91
Intercept (c): 2.02
```

So the **prediction equation** becomes:

$$
\text{Score} = 9.91 \times \text{Hours} + 2.02
$$

---


