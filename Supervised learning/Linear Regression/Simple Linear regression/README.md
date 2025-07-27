
---

## ✅ 1. **Overview of Linear Regression**

Linear Regression is a **supervised machine learning** algorithm used to model the **linear relationship** between a **dependent variable (Y)** and one or more **independent variables (X)**.

In **simple linear regression**, the equation of the line is:

$$
\boxed{y = mx + c}
$$

Where:

* $y$: predicted value
* $x$: input feature (independent variable)
* $m$: slope of the line (rate of change)
* $c$: y-intercept (value of y when x = 0)

---

## ✅ 2. **From Image 1: Visual Concepts**

### 🔹 Components Explained:

* **Orange dots**: Actual data (observed values of y)
* **Blue dashed line**: Regression line (predicted y)
* **Black dashed lines**: **Residuals** (errors) = actual $y$ − predicted $y$
* **$\Delta y / \Delta x$**: Rise over run = slope $m$
* **Starting Price → Ending Price**: Represents X to Y mapping
* **Y-intercept**: Where line crosses Y-axis when $x = 0$

### 🔹 Slope Calculation (Visual):

$$
m = \frac{\Delta y}{\Delta x}
$$

This gives the **rate of change** of Y with respect to X.

---

## ✅ 3. **From Image 2: Mathematical Formulas**

### 🔹 Slope Formula (Least Squares Estimation):

$$
\boxed{
m = \frac{\sum (x - \bar{x})(y - \bar{y})}{\sum (x - \bar{x})^2}
}
$$

Where:

* $\bar{x}$ = mean of x values
* $\bar{y}$ = mean of y values
* Numerator = how x and y vary together (covariance)
* Denominator = how x varies (variance of x)

---

### 🔹 Intercept Formula:

$$
\boxed{c = \bar{y} - m\bar{x}}
$$

It adjusts the line vertically to fit the data correctly.

---

### 🔹 Final Prediction Equation:

$$
\boxed{\text{Prediction}_y = m \cdot (\text{input}_X) + c}
$$

---

## ✅ 4. **Graph (Image 2) Interpretation**

* Blue dots = data points (actual)
* Dashed green line = regression line
* Red triangle = visual representation of slope $m = \frac{\Delta y}{\Delta x}$
* Point where the line touches Y-axis = intercept $c$

---

## ✅ 5. **What Linear Regression Does**

* Tries to draw a **straight line** that minimizes the **error (residual)** between actual and predicted values.
* Uses **least squares method** to minimize total squared error:

  $$
  \text{Minimize } \sum (y_i - \hat{y}_i)^2
  $$

---

## ✅ 6. **Key Concepts Summary**

| Term                | Meaning                                   |
| ------------------- | ----------------------------------------- |
| **m** (slope)       | Change in Y per unit change in X          |
| **c** (intercept)   | Predicted Y when X = 0                    |
| **Residual**        | Difference between actual and predicted Y |
| **Regression Line** | Best-fit line minimizing squared errors   |
| **Objective**       | Minimize total residual squared error     |

---

