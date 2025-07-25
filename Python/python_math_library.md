
---

## 📚 Full Documentation of `math` Library in Python

### 🔹 Import:

```python
import math
```

---

## 🔢 1. Constants (স্থায়ী মান)

| Constant   | Description                  | Example    |
| ---------- | ---------------------------- | ---------- |
| `math.pi`  | π ≈ 3.14159                  | `math.pi`  |
| `math.e`   | e ≈ 2.71828 (Euler's number) | `math.e`   |
| `math.tau` | τ = 2π                       | `math.tau` |
| `math.inf` | Infinity                     | `math.inf` |
| `math.nan` | Not a Number                 | `math.nan` |

---

## 🧮 2. Basic Math Functions

| Function            | Description                         | Example                   |
| ------------------- | ----------------------------------- | ------------------------- |
| `math.ceil(x)`      | Smallest integer ≥ x                | `math.ceil(4.3)` → 5      |
| `math.floor(x)`     | Largest integer ≤ x                 | `math.floor(4.7)` → 4     |
| `math.trunc(x)`     | Truncates decimal (like int())      | `math.trunc(4.7)` → 4     |
| `math.fabs(x)`      | Absolute value (always float)       | `math.fabs(-4)` → 4.0     |
| `math.factorial(x)` | x! (non-negative integers only)     | `math.factorial(5)` → 120 |
| `math.gcd(a, b)`    | Greatest common divisor             | `math.gcd(12, 8)` → 4     |
| `math.lcm(a, b)`    | Least common multiple (Python 3.9+) | `math.lcm(3, 4)` → 12     |
| `math.isqrt(n)`     | Integer square root                 | `math.isqrt(10)` → 3      |

---

## 🧠 3. Power & Logarithmic Functions

| Function            | Description           | Example                 |
| ------------------- | --------------------- | ----------------------- |
| `math.pow(x, y)`    | x raised to y (float) | `math.pow(2, 3)` → 8.0  |
| `math.exp(x)`       | eˣ                    | `math.exp(2)` → 7.389   |
| `math.log(x)`       | Natural log (base e)  | `math.log(10)`          |
| `math.log(x, base)` | Log base n            | `math.log(100, 10)` → 2 |
| `math.log10(x)`     | Log base 10           | `math.log10(100)` → 2   |
| `math.log2(x)`      | Log base 2            | `math.log2(8)` → 3      |
| `math.sqrt(x)`      | Square root           | `math.sqrt(16)` → 4.0   |

---

## 📐 4. Trigonometric Functions (Radians-based)

| Function           | Description                | Example                   |
| ------------------ | -------------------------- | ------------------------- |
| `math.sin(x)`      | Sine                       | `math.sin(math.pi/2)` → 1 |
| `math.cos(x)`      | Cosine                     | `math.cos(0)` → 1         |
| `math.tan(x)`      | Tangent                    | `math.tan(math.pi/4)` → 1 |
| `math.asin(x)`     | Arc sine (returns radians) | `math.asin(1)` → π/2      |
| `math.acos(x)`     | Arc cosine                 | `math.acos(0)` → π/2      |
| `math.atan(x)`     | Arc tangent                | `math.atan(1)` → π/4      |
| `math.atan2(y, x)` | Arc tangent of y/x         | `math.atan2(1, 1)` → π/4  |
| `math.hypot(x, y)` | √(x² + y²)                 | `math.hypot(3, 4)` → 5.0  |

---

## 🔁 5. Degree & Radian Conversion

| Function          | Description       | Example                       |
| ----------------- | ----------------- | ----------------------------- |
| `math.degrees(x)` | Radians → Degrees | `math.degrees(math.pi)` → 180 |
| `math.radians(x)` | Degrees → Radians | `math.radians(180)` → π       |

---

## 🧊 6. Special Functions

| Function              | Description                                 | Example                       |
| --------------------- | ------------------------------------------- | ----------------------------- |
| `math.isfinite(x)`    | Checks if x is neither `inf` nor `NaN`      | `math.isfinite(10)` → True    |
| `math.isinf(x)`       | Is x infinite?                              | `math.isinf(math.inf)` → True |
| `math.isnan(x)`       | Is x NaN (not a number)?                    | `math.isnan(math.nan)` → True |
| `math.copysign(x, y)` | Returns x with the sign of y                | `math.copysign(2, -3)` → -2.0 |
| `math.modf(x)`        | Returns fractional & integer parts as tuple | `math.modf(3.14)` → (0.14, 3) |
| `math.frexp(x)`       | Breaks float into mantissa and exponent     | `math.frexp(8.0)` → (0.5, 4)  |
| `math.ldexp(x, i)`    | x × (2^i)                                   | `math.ldexp(0.5, 3)` → 4.0    |

---

## ✅ Example Usage:

```python
import math

print("π =", math.pi)
print("Square root of 25:", math.sqrt(25))
print("Sin(90°):", math.sin(math.radians(90)))
print("Factorial of 5:", math.factorial(5))
```

---

## 🔗 Official Python Documentation:

If you want the latest and most accurate reference:
📎 [https://docs.python.org/3/library/math.html](https://docs.python.org/3/library/math.html)

---


---

## 🧠 Problem Set Using `math` Library

---

### ✅ **1. Find the Area of a Circle**

**Problem**: Given a radius `r`, calculate the area of the circle.
**Formula**: `Area = π × r²`

```python
import math

r = 5
area = math.pi * math.pow(r, 2)
print("Area of Circle:", area)
```

---

### ✅ **2. Calculate the Hypotenuse**

**Problem**: Given two sides `a` and `b` of a right triangle, find the hypotenuse.
**Formula**: `c = √(a² + b²)`

```python
import math

a = 3
b = 4
c = math.hypot(a, b)
print("Hypotenuse:", c)
```

---

### ✅ **3. Convert Degrees to Radians**

**Problem**: Convert 180 degrees to radians.

```python
import math

degrees = 180
radians = math.radians(degrees)
print("180 degrees in radians:", radians)
```

---

### ✅ **4. Compute Logarithm with Base 10**

**Problem**: Find log base 10 of a number.

```python
import math

num = 1000
log_val = math.log10(num)
print("log10 of", num, "is", log_val)
```

---

### ✅ **5. Find Factorial of a Number**

**Problem**: Calculate the factorial of 6.

```python
import math

n = 6
fact = math.factorial(n)
print("Factorial of", n, "is", fact)
```

---

### ✅ **6. Round a Decimal Up and Down**

**Problem**: Given a float number, round it up and down.

```python
import math

num = 4.7
print("Ceil:", math.ceil(num))   # 5
print("Floor:", math.floor(num)) # 4
```

---

### ✅ **7. Solve for Power and Square Root**

**Problem**: Calculate 2⁸ and √64.

```python
import math

power_val = math.pow(2, 8)
sqrt_val = math.sqrt(64)
print("2^8 =", power_val)
print("Square root of 64:", sqrt_val)
```

---

### ✅ **8. Check if a Value is Infinite or NaN**

**Problem**: Use `math.inf` and `math.nan` in conditions.

```python
import math

x = math.inf
y = math.nan

print("Is x infinite?", math.isinf(x))
print("Is y NaN?", math.isnan(y))
```

---

### ✅ **9. Calculate Angle Using `atan2()`**

**Problem**: Find the angle (in degrees) of a vector with components (y=5, x=5).

```python
import math

x = 5
y = 5
angle_rad = math.atan2(y, x)
angle_deg = math.degrees(angle_rad)
print("Angle in degrees:", angle_deg)
```

---

### ✅ **10. Find LCM and GCD**

**Problem**: Given two numbers, find their LCM and GCD.

```python
import math

a = 15
b = 20

gcd_val = math.gcd(a, b)
lcm_val = math.lcm(a, b)  # Python 3.9+

print("GCD:", gcd_val)
print("LCM:", lcm_val)
```

---

---

## 🧪 Math-Based Python Interview Problems with Solutions

---

### ✅ **Problem 1: Check if a Number is Power of Two**

**Statement**: Write a function to check if a number `n` is a power of 2.

```python
import math

def is_power_of_two(n):
    if n <= 0:
        return False
    return math.log2(n).is_integer()

# Test
print(is_power_of_two(16))  # True
print(is_power_of_two(18))  # False
```

---

### ✅ **Problem 2: Count Digits in a Factorial**

**Statement**: For a given number `n`, find the number of digits in `n!`.

**Hint**: Use `log10(n!) = log10(1) + log10(2) + ... + log10(n)`

```python
import math

def count_digits_in_factorial(n):
    if n == 0 or n == 1:
        return 1
    digits = 0
    for i in range(2, n + 1):
        digits += math.log10(i)
    return math.floor(digits) + 1

# Test
print(count_digits_in_factorial(5))   # Output: 3 (120)
print(count_digits_in_factorial(100)) # Output: 158
```

---

### ✅ **Problem 3: Closest Power of 2 Less Than or Equal to N**

```python
import math

def closest_power_of_2(n):
    return 2 ** math.floor(math.log2(n))

# Test
print(closest_power_of_2(20))  # Output: 16
```

---

### ✅ **Problem 4: Find Distance Between Two Points**

**Statement**: Given 2 points `(x1, y1)` and `(x2, y2)`, find the Euclidean distance.

```python
import math

def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

# Test
print(distance(0, 0, 3, 4))  # Output: 5.0
```

---

### ✅ **Problem 5: Find nth Root of a Number**

**Statement**: Given two numbers `x` and `n`, find the nth root of x.

```python
import math

def nth_root(x, n):
    return math.pow(x, 1/n)

# Test
print(nth_root(27, 3))  # Output: 3.0
```

---

### ✅ **Problem 6: Check if Point is Inside a Circle**

```python
import math

def is_inside_circle(circle_x, circle_y, radius, point_x, point_y):
    dist = math.hypot(point_x - circle_x, point_y - circle_y)
    return dist <= radius

# Test
print(is_inside_circle(0, 0, 5, 2, 2))  # True
print(is_inside_circle(0, 0, 5, 6, 0))  # False
```

---

### ✅ **Problem 7: Angle Between Hour and Minute Hands**

**Statement**: Given time in hour and minutes, find the angle between clock hands.

```python
import math

def clock_angle(hour, minute):
    hour %= 12
    hour_angle = (hour * 30) + (minute * 0.5)
    minute_angle = minute * 6
    angle = abs(hour_angle - minute_angle)
    return min(angle, 360 - angle)

# Test
print(clock_angle(3, 30))  # Output: 75.0
```

---

### ✅ **Problem 8: Check for Perfect Square**

```python
import math

def is_perfect_square(n):
    root = math.isqrt(n)
    return root * root == n

# Test
print(is_perfect_square(49))  # True
print(is_perfect_square(50))  # False
```

---

### ✅ **Problem 9: Sum of Digits in Factorial**

```python
import math

def sum_of_digits_in_factorial(n):
    fact = math.factorial(n)
    return sum(int(d) for d in str(fact))

# Test
print(sum_of_digits_in_factorial(10))  # Output: 27 (3628800)
```

---

### ✅ **Problem 10: Check if Angle is Acute, Right or Obtuse**

```python
import math

def classify_angle(a, b, c):
    # Using cosine rule
    cos_C = (a**2 + b**2 - c**2) / (2 * a * b)
    angle = math.degrees(math.acos(cos_C))

    if angle < 90:
        return "Acute"
    elif angle == 90:
        return "Right"
    else:
        return "Obtuse"

# Test
print(classify_angle(3, 4, 5))  # Output: Right
```

---



