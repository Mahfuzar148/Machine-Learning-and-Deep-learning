

---

# ✅ **Project 4: Salary Prediction Based on Experience**

**Goal**: Use **Years of Experience** to predict **Salary** with simple linear regression.

**Dataset**:
[🔗 Salary Dataset (YearsExperience vs Salary)](https://www.kaggle.com/datasets/karthickveerakumar/salary-data-simple-linear-regression)

---

## 🔧 Step-by-Step Implementation

---

### ✅ Step 1: Install Required Libraries

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

### ✅ Step 2: Load the Dataset

```python
import pandas as pd

# Load CSV file (download from Kaggle and place in working directory)
df = pd.read_csv("Salary_Data.csv")

# View top rows
print(df.head())
```

The dataset contains:

* `YearsExperience`: Number of years worked
* `Salary`: Annual salary

---

### ✅ Step 3: Prepare the Data

```python
X = df[['YearsExperience']]  # 2D
y = df['Salary']             # 1D
```

---

### ✅ Step 4: Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### ✅ Step 5: Train Linear Regression Model

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

---

### ✅ Step 6: Evaluate the Model

```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)

print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
```

---

### ✅ Step 7: Plot the Regression Line

```python
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='green', label='Actual Salary')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Experience vs Salary Prediction')
plt.legend()
plt.grid(True)
plt.show()
```

---

### 🏁 Example Output

```
Slope (m): 9449.96
Intercept (c): 25792.20
R² Score: 0.96
```

### ✅ Interpretation

* For every 1 year of experience, salary increases by about \$9,450.
* The model fits very well (R² = 0.96 → 96% of variance explained).

---

Would you like to continue with:

| Next Topic Idea                  | Description                                                           |
| -------------------------------- | --------------------------------------------------------------------- |
| **5. Multi-variable Regression** | Predict salary using multiple features (e.g., education + experience) |
| **6. Polynomial Regression**     | Handle curved relationships                                           |
| **7. Custom Dataset**            | Upload your own and build a project together                          |


