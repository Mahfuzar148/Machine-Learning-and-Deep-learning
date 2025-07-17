

## 📘 Table of Contents

1. [📌 Variables](#📌-variables)
2. [📌 Data Types](#📌-data-types)
3. [📌 Type Casting](#📌-type-casting)
4. [📌 Input and Output](#📌-input-and-output)
5. [🧪 Practice Examples (25+)](#🧪-practice-examples-25)

---

## 📌 Variables

### ✅ What is a Variable?

A **variable** is a name that refers to a value stored in memory.

### ✅ Rules for Naming Variables

* Must begin with a letter (a–z, A–Z) or an underscore (\_)
* Cannot start with a number
* Cannot be a keyword (like `if`, `while`, `class`, etc.)

### ✅ Examples

```python
x = 10
name = "Alice"
_is_active = True
```

---

## 📌 Data Types

### Python has several built-in data types:

| Category | Type                               | Example                  |
| -------- | ---------------------------------- | ------------------------ |
| Numeric  | `int`, `float`, `complex`          | 5, 3.14, 2+3j            |
| Text     | `str`                              | "Hello", 'A'             |
| Boolean  | `bool`                             | `True`, `False`          |
| Sequence | `list`, `tuple`, `range`           | \[1,2], (1,2), range(5)  |
| Set      | `set`, `frozenset`                 | {1,2}, frozenset(\[1,2]) |
| Mapping  | `dict`                             | {"a": 1, "b": 2}         |
| Binary   | `bytes`, `bytearray`, `memoryview` | `b"hello"`               |
| None     | `NoneType`                         | `None`                   |

### ✅ Examples

```python
age = 25                    # int
pi = 3.14                   # float
name = "David"              # str
is_valid = True             # bool
fruits = ["apple", "banana"] # list
person = {"name": "Tom", "age": 30} # dict
```

---

## 📌 Type Casting

### ✅ Converting one type to another:

```python
x = int("10")      # from string to int
y = float(5)       # from int to float
z = str(3.14)      # from float to string
```

---

## 📌 Input and Output

### ✅ Input from user

```python
name = input("Enter your name: ")
```

> *Note: `input()` always returns a string.*

### ✅ Output using `print()`

```python
print("Hello, world!")
print("Your name is", name)
print(f"Welcome, {name}")  # f-string formatting
```

---

## 🧪 Practice Examples (25+)

Here are 25+ coding exercises for practice. You can copy-paste and modify them to test your understanding.

---

### 🔢 1. Add two numbers

```python
a = int(input("Enter first number: "))
b = int(input("Enter second number: "))
print("Sum is:", a + b)
```

### 🔄 2. Swap two variables

```python
x = 5
y = 10
x, y = y, x
print("x:", x, "y:", y)
```

### 🔍 3. Check if a number is even or odd

```python
num = int(input("Enter a number: "))
print("Even" if num % 2 == 0 else "Odd")
```

### 📅 4. Get age from birth year

```python
birth_year = int(input("Enter birth year: "))
current_year = 2025
print("Your age is:", current_year - birth_year)
```

### 🔤 5. Reverse a string

```python
s = input("Enter a string: ")
print("Reversed:", s[::-1])
```

### 🔡 6. Convert lowercase to uppercase

```python
s = input("Enter lowercase string: ")
print("Uppercase:", s.upper())
```

### 🧮 7. Find square and cube

```python
n = int(input("Enter a number: "))
print("Square:", n**2, "Cube:", n**3)
```

### 💯 8. Percentage calculator

```python
total = int(input("Total marks: "))
obtained = int(input("Obtained marks: "))
percentage = (obtained / total) * 100
print("Percentage:", percentage)
```

### 🔠 9. Check character type

```python
ch = input("Enter a character: ")
if ch.isalpha():
    print("Alphabet")
elif ch.isdigit():
    print("Digit")
else:
    print("Special character")
```

### 🔄 10. Celsius to Fahrenheit

```python
c = float(input("Enter Celsius: "))
f = (c * 9/5) + 32
print("Fahrenheit:", f)
```


---

### 11. ✅ Check if number is positive or negative

```python
num = float(input("Enter a number: "))
if num > 0:
    print("Positive")
elif num < 0:
    print("Negative")
else:
    print("Zero")
```

---

### 12. ✅ Count vowels in a string

```python
s = input("Enter a string: ")
vowels = "aeiouAEIOU"
count = 0
for ch in s:
    if ch in vowels:
        count += 1
print("Number of vowels:", count)
```

---

### 13. ✅ Find max of 3 numbers

```python
a = int(input("Enter first number: "))
b = int(input("Enter second number: "))
c = int(input("Enter third number: "))
print("Maximum is:", max(a, b, c))
```

---

### 14. ✅ Area of circle

```python
import math
r = float(input("Enter radius: "))
area = math.pi * r * r
print("Area of circle:", area)
```

---

### 15. ✅ Simple calculator (menu-based)

```python
print("1. Add\n2. Subtract\n3. Multiply\n4. Divide")
choice = input("Choose operation (1-4): ")

a = float(input("Enter first number: "))
b = float(input("Enter second number: "))

if choice == "1":
    print("Result:", a + b)
elif choice == "2":
    print("Result:", a - b)
elif choice == "3":
    print("Result:", a * b)
elif choice == "4":
    if b != 0:
        print("Result:", a / b)
    else:
        print("Cannot divide by zero!")
else:
    print("Invalid choice")
```

---

### 16. ✅ Check leap year

```python
year = int(input("Enter year: "))
if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
    print("Leap year")
else:
    print("Not a leap year")
```

---

### 17. ✅ Print first n natural numbers

```python
n = int(input("Enter a number: "))
for i in range(1, n+1):
    print(i, end=' ')
```

---

### 18. ✅ Sum of digits of a number

```python
num = int(input("Enter a number: "))
sum_digits = 0
while num > 0:
    sum_digits += num % 10
    num //= 10
print("Sum of digits:", sum_digits)
```

---

### 19. ✅ Palindrome check

```python
s = input("Enter a word: ")
if s == s[::-1]:
    print("Palindrome")
else:
    print("Not a palindrome")
```

---

### 20. ✅ Multiplication table

```python
n = int(input("Enter a number: "))
for i in range(1, 11):
    print(f"{n} x {i} = {n*i}")
```

---

### 21. ✅ BMI Calculator

```python
weight = float(input("Enter weight in kg: "))
height = float(input("Enter height in meters: "))
bmi = weight / (height ** 2)
print("BMI:", round(bmi, 2))
```

---

### 22. ✅ Check if a string is numeric

```python
s = input("Enter something: ")
if s.isnumeric():
    print("It is numeric.")
else:
    print("It is not numeric.")
```

---

### 23. ✅ Type checking using `type()`

```python
x = input("Enter something: ")

if x.isdigit():
    x = int(x)
elif '.' in x:
    try:
        x = float(x)
    except:
        pass

print("Type is:", type(x))
```

---

### 24. ✅ Convert float to int and compare

```python
f = float(input("Enter a float number: "))
i = int(f)
print("Integer:", i)
if f == i:
    print("Equal")
else:
    print("Not equal")
```

---

### 25. ✅ Check if input is even-length string

```python
s = input("Enter a string: ")
if len(s) % 2 == 0:
    print("Even length")
else:
    print("Odd length")
```

---



### 26. ✅ Print all even numbers between 1 and n

```python
n = int(input("Enter the limit: "))
for i in range(1, n+1):
    if i % 2 == 0:
        print(i, end=' ')
```

---

### 27. ✅ Count digits in a number

```python
num = int(input("Enter a number: "))
count = 0
while num > 0:
    count += 1
    num //= 10
print("Number of digits:", count)
```

---

### 28. ✅ Count words in a sentence

```python
sentence = input("Enter a sentence: ")
words = sentence.split()
print("Total words:", len(words))
```

---

### 29. ✅ Print ASCII value of a character

```python
ch = input("Enter a character: ")
print("ASCII value of", ch, "is", ord(ch))
```

---

### 30. ✅ Find factorial of a number

```python
n = int(input("Enter a number: "))
fact = 1
for i in range(1, n+1):
    fact *= i
print("Factorial:", fact)
```

---

### 31. ✅ Check if number is prime

```python
num = int(input("Enter a number: "))
is_prime = True
if num < 2:
    is_prime = False
else:
    for i in range(2, int(num**0.5)+1):
        if num % i == 0:
            is_prime = False
            break
print("Prime" if is_prime else "Not prime")
```

---

### 32. ✅ Print Fibonacci series (first n terms)

```python
n = int(input("Enter number of terms: "))
a, b = 0, 1
for _ in range(n):
    print(a, end=' ')
    a, b = b, a + b
```

---

### 33. ✅ Count uppercase letters in a string

```python
s = input("Enter a string: ")
count = sum(1 for c in s if c.isupper())
print("Uppercase letters:", count)
```

---

### 34. ✅ Find average of n numbers

```python
n = int(input("How many numbers: "))
total = 0
for i in range(n):
    num = float(input("Enter number: "))
    total += num
print("Average:", total / n)
```

---

### 35. ✅ Check if year is century year

```python
year = int(input("Enter year: "))
print("Century year" if year % 100 == 0 else "Not a century year")
```

---

### 36. ✅ Count characters in string without spaces

```python
s = input("Enter a string: ")
count = len(s.replace(" ", ""))
print("Characters without spaces:", count)
```

---

### 37. ✅ Convert seconds into hours, minutes, and seconds

```python
total_seconds = int(input("Enter seconds: "))
hours = total_seconds // 3600
minutes = (total_seconds % 3600) // 60
seconds = total_seconds % 60
print(f"{hours}h {minutes}m {seconds}s")
```

---

### 38. ✅ Check if number is Armstrong (3-digit)

```python
num = int(input("Enter a 3-digit number: "))
sum_cubes = sum(int(d)**3 for d in str(num))
print("Armstrong" if sum_cubes == num else "Not Armstrong")
```

---

### 39. ✅ Convert number to binary

```python
num = int(input("Enter a number: "))
print("Binary:", bin(num)[2:])
```

---

### 40. ✅ Convert binary to decimal

```python
binary = input("Enter binary number: ")
print("Decimal:", int(binary, 2))
```

---

### 41. ✅ Find largest digit in a number

```python
num = input("Enter a number: ")
print("Largest digit:", max(num))
```

---

### 42. ✅ Check if string contains only alphabets

```python
s = input("Enter a string: ")
print("Only alphabets" if s.isalpha() else "Contains other characters")
```

---

### 43. ✅ Count specific character in string

```python
s = input("Enter a string: ")
ch = input("Character to count: ")
print(f"{ch} appears {s.count(ch)} times")
```

---

### 44. ✅ Check if string is all lowercase

```python
s = input("Enter a string: ")
print("All lowercase" if s.islower() else "Not all lowercase")
```

---

### 45. ✅ Find sum of odd numbers from 1 to n

```python
n = int(input("Enter a number: "))
total = sum(i for i in range(1, n+1) if i % 2 != 0)
print("Sum of odd numbers:", total)
```

---

### 46. ✅ Add two numbers without using `+`

```python
a = int(input("Enter first number: "))
b = int(input("Enter second number: "))
import operator
print("Sum is:", operator.add(a, b))
```

---

### 47. ✅ Find median of 3 numbers

```python
a = int(input("Enter first: "))
b = int(input("Enter second: "))
c = int(input("Enter third: "))
print("Median:", sorted([a, b, c])[1])
```

---

### 48. ✅ Convert string to list

```python
s = input("Enter comma-separated items: ")
items = s.split(",")
print("List:", items)
```

---

### 49. ✅ Count spaces in a string

```python
s = input("Enter a string: ")
print("Spaces:", s.count(" "))
```

---

### 50. ✅ Capitalize first letter of each word

```python
s = input("Enter a sentence: ")
print("Title case:", s.title())
```

---


## 🧠 Critical Problem List with Solutions

### ✅ Problem 1: **Detect Data Type of Each Word in a String**

**Description:**
Take an input string where words can be digits, floats, or text. Identify the data type of each word.

**Example:**

```
Input: "42 3.14 hello 999 world 5.0"
Output: int float str int str float
```

**Solution:**

```python
data = input("Enter words: ").split()
for item in data:
    try:
        val = int(item)
        print("int", end=' ')
    except ValueError:
        try:
            val = float(item)
            print("float", end=' ')
        except ValueError:
            print("str", end=' ')
```

---

### ✅ Problem 2: **Valid Integer Without Using `isdigit()`**

**Description:**
Check if a given string is a valid integer, handling negative numbers and not using `isdigit()`.

```python
def is_integer(s):
    if s.startswith('-'):
        s = s[1:]
    print("Valid Integer" if s and all('0' <= ch <= '9' for ch in s) else "Invalid")
    
is_integer(input("Enter input: "))
```

---

### ✅ Problem 3: **Custom Calculator with Mixed Input**

**Description:**
Take input as a string like `"12 + 7.5"` and perform the operation without using `eval()`.

```python
expr = input("Enter expression (a op b): ")  # e.g., 12 + 7.5
a, op, b = expr.split()
a = float(a)
b = float(b)

if op == '+': print("Result:", a + b)
elif op == '-': print("Result:", a - b)
elif op == '*': print("Result:", a * b)
elif op == '/': print("Result:", a / b if b != 0 else "Undefined")
else: print("Invalid Operator")
```

---

### ✅ Problem 4: **Decode Mixed Input into Cleaned Data Dictionary**

**Input:** A string input in format like:
`"name:John age:25 height:5.9 active:True"`

**Goal:** Convert into a dictionary with correct types.

```python
raw = input("Enter data: ").split()
data = {}
for pair in raw:
    key, val = pair.split(':')
    if val.isdigit():
        val = int(val)
    elif '.' in val:
        try:
            val = float(val)
        except:
            pass
    elif val in ['True', 'False']:
        val = val == 'True'
    data[key] = val

print(data)
```

---

### ✅ Problem 5: **Type-Sensitive Sorting**

**Description:**
Given a mixed list of values, sort integers, floats, and strings separately but keep their types.

```python
mixed = ['5.5', '8', 'abc', '12.0', '4', 'hello', '1']
ints = []
floats = []
strings = []

for item in mixed:
    try:
        val = int(item)
        ints.append(val)
    except:
        try:
            val = float(item)
            floats.append(val)
        except:
            strings.append(item)

print("Integers:", sorted(ints))
print("Floats:", sorted(floats))
print("Strings:", sorted(strings))
```

---

### ✅ Problem 6: **Validate Manual Type Conversion and Print Error Info**

```python
val = input("Enter something: ")
try:
    print("As int:", int(val))
except Exception as e:
    print("Cannot convert to int:", e)

try:
    print("As float:", float(val))
except Exception as e:
    print("Cannot convert to float:", e)
```

---

### ✅ Problem 7: **Input Parser: Name & Age Validation with Output**

**Input format:** `"name:John age:twenty"`

Validate and print meaningful messages.

```python
entry = input("Enter name and age (e.g. name:John age:30): ")
parts = dict(p.split(':') for p in entry.split())

name = parts.get("name")
age = parts.get("age")

if name and age:
    try:
        age = int(age)
        print(f"Hello {name}, your age is {age}")
    except:
        print("Age must be a number.")
else:
    print("Invalid input format.")
```

---

### ✅ Problem 8: **Smart Converter: User Inputs Amount and Unit**

**Input:** `"15 cm"`
**Goal:** Convert to inches, meters, etc.

```python
val = input("Enter amount and unit (e.g. 15 cm): ").split()
amount = float(val[0])
unit = val[1].lower()

if unit == "cm":
    print(f"{amount} cm = {amount / 2.54:.2f} inches")
elif unit == "inch":
    print(f"{amount} inches = {amount * 2.54:.2f} cm")
else:
    print("Unsupported unit.")
```

---

### ✅ Problem 9: **Boolean Evaluator From String Input**

```python
val = input("Enter boolean value (True/False): ").strip().lower()
if val in ['true', '1', 'yes']:
    result = True
elif val in ['false', '0', 'no']:
    result = False
else:
    result = None

print("Boolean interpreted as:", result)
```

---

### ✅ Problem 10: **Multi-Type Input Handler for Tabular Input**

**Input format:**

```
3
John,25,5.9
Alice,22,5.5
Mark,27,6.1
```

Print all names where height > 5.8.

```python
n = int(input("How many entries? "))
for _ in range(n):
    name, age, height = input().split(',')
    if float(height) > 5.8:
        print(name)
```

---

## 📚 Sources & Inspirations

These problems are inspired by:

* [Python official docs](https://docs.python.org/3/)
* HackerRank (Input/Output challenges)
* Codewars and LeetCode basics
* Real-world Python scripting tasks (type handling, input validation)

---



Below is a **curated “critical‑thinking” problem set (20 tasks)** that centres on **variables, data types, and input/output in Python**.
For every problem you will find:

* **Source** – where the original idea/challenge comes from.
* **What to practise** – the key concept(s) being exercised.
* **Task** – a concise description.
* **Illustrative solution** – clean, idiomatic Python 3 code (no external libraries unless the source itself requires one).

---

### 1 String Formatting Across Bases (HackerRank) ([HackerRank][1])

**Practise:** numeric types, f‑strings / `format()`.
**Task:** For `n`, print numbers `1…n` left‑aligned in decimal, octal, hex (uppercase) and binary.

```python
def print_formatted(n: int) -> None:
    w = len(bin(n)) - 2          # width of the binary column
    for i in range(1, n + 1):
        print(f"{i:{w}d} {i:{w}o} {i:{w}X} {i:{w}b}")
```

---

### 2 Alphabet Rangoli (HackerRank) ([HackerRank][2])

**Practise:** string slicing/joins, centred I/O formatting.
**Task:** Draw a “rangoli” letter pattern of size `n`.

```python
import string

def rangoli(n: int) -> str:
    alpha = string.ascii_lowercase
    lines = []
    for i in range(n):
        s = "-".join(alpha[i:n])
        lines.append((s[::-1] + s[1:]).center(4*n-3, "-"))
    return "\n".join(lines[::-1] + lines[1:])
```

---

### 3 Validate a Number String (LeetCode #65) ([LeetCode][3])

**Practise:** regex vs. manual parsing, defensive input checks.
**Task:** Return `True` if `s` is a valid int/float with optional sign & exponent.

```python
import re

def is_number(s: str) -> bool:
    pattern = re.compile(r"""
        ^[+-]?(
            (\d+\.\d*) | (\.\d+) | (\d+)
        )
        ([eE][+-]?\d+)?$
    """, re.VERBOSE)
    return bool(pattern.match(s.strip()))
```

---

### 4 Parse Float (Codewars) ([Codewars][4])

**Practise:** safe casting, exception handling.

```python
def parse_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
```

---

### 5 Extract Age From String (Codewars) ([Codewars][5])

**Practise:** indexing, ASCII to int.

```python
def girl_age(txt: str) -> int:
    return int(txt[0])          # first char is always the digit
```

---

### 6 Armstrong / Narcissistic Number (w3resource) ([w3resource][6])

**Practise:** numeric ops, `str` ⇆ `int` conversion.

```python
def is_armstrong(num: int) -> bool:
    k = len(str(num))
    return num == sum(int(d)**k for d in str(num))
```

---

\### 7 Sum‑Square Difference (Project Euler #6) ([zach.se][7])
**Practise:** arithmetic series, ints vs. big ints.

```python
def sum_square_diff(n=100):
    return (sum(range(1, n+1))**2) - sum(i*i for i in range(1, n+1))
```

---

\### 8 Token‑Type Classifier (PyNative) ([PYnative][8])
**Practise:** dynamic typing, chained `try/except`.

```python
def classify(tokens: str) -> list[str]:
    out = []
    for t in tokens.split():
        try:
            int(t); out.append("int")
        except ValueError:
            try:
                float(t); out.append("float")
            except ValueError:
                out.append("str")
    return out
```

---

\### 9 Read Until EOF (StackOverflow) ([Stack Overflow][9])
**Practise:** robust stdin processing.

```python
import sys
total = 0
for line in sys.stdin:          # stops gracefully at EOF
    if line.strip():            # skip blank lines
        total += int(line)
print(total)
```

---

\### 10 Least‑Frequent Character (GeeksforGeeks) ([GeeksforGeeks][10])
**Practise:** `collections.Counter`.

```python
from collections import Counter
def least_freq_char(s: str) -> str:
    return min(Counter(s), key=Counter(s).get)
```

---

\### 11 Sort a Mixed‑Type List (GeeksforGeeks) ([GeeksforGeeks][11])
**Practise:** custom key functions, type inspection.

```python
def smart_sort(lst):
    ints   = sorted(i   for i in lst if isinstance(i, int))
    floats = sorted(f   for f in lst if isinstance(f, float))
    strs   = sorted(s   for s in lst if isinstance(s, str))
    return ints + floats + strs
```

---

\### 12 Seconds → `hh:mm:ss` (w3resource) ([w3resource][12])

```python
def hms(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02}:{m:02}:{s:02}"
```

---

\### 13 Count Vowels in Specific Lines (PyNative) ([PYnative][13])
**Practise:** file I/O, slicing, generators.

```python
def vowel_count(path, start=1, end=5):
    vowels = "aeiouAEIOU"
    with open(path, encoding="utf-8") as f:
        seg = (line for idx, line in enumerate(f, 1) if start <= idx <= end)
        return sum(ch in vowels for line in seg for ch in line)
```

---

\### 14 Distinct Stamps with `set.add()` (HackerRank) ([HackerRank][14])

```python
def unique_countries(lines: list[str]) -> int:
    seen = set()
    for c in lines:
        seen.add(c.strip())
    return len(seen)
```

---

\### 15 Symmetric Difference of Sets (HackerRank) ([HackerRank][15])

```python
def sym_diff(a: set[int], b: set[int]) -> list[int]:
    return sorted(a ^ b)
```

---

\### 16 Email Validator (GeeksforGeeks) ([GeeksforGeeks][16])

```python
import re
EMAIL_RE = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w{2,}$")
is_valid_email = lambda e: bool(EMAIL_RE.match(e))
```

---

\### 17 Mini CLI with `argparse` (Real Python) ([Real Python][17])
**Practise:** typed command‑line input.

```python
import argparse, statistics

def main():
    p = argparse.ArgumentParser(description="Compute stats.")
    p.add_argument("nums", type=float, nargs="+")
    args = p.parse_args()
    print("mean:", statistics.mean(args.nums))
    print("stdev:", statistics.stdev(args.nums))

if __name__ == "__main__":
    main()
```

---

\### 18 Decimal → Binary via Recursion (TutorialsPoint) ([TutorialsPoint][18])

```python
def dec2bin(n: int) -> str:
    return "0" if n == 0 else dec2bin(n // 2).lstrip("0") + str(n % 2)
```

---

\### 19 Explicit Type Conversion (Programiz) ([Programiz][19])

```python
def safe_sum(a: str, b: str) -> int:
    return int(a) + int(b)          # raises if inputs not numeric
```

---

\### 20 Numeric Tower Comparison (Python Docs) ([Python documentation][20])
**Practise:** mixed‑type arithmetic & comparisons.

```python
def compare_demo(x, y):
    print(f"{x!r} ({type(x).__name__})  vs  {y!r} ({type(y).__name__})")
    print("x == y :", x == y)
    print("x <  y :", x <  y)
    print("x + y  :", x + y)
```

---

## How to Use This List

1. **Work through each problem** without immediately looking at the sample code.
2. **Run the snippets**, tweak inputs, and add edge‑cases until the behaviour is crystal‑clear.
3. **Refactor**: can you shorten, speed‑up, or generalise the solution?
4. **Translate** any exercise into Bengali (or another language) if that aids understanding.


[1]: https://www.hackerrank.com/challenges/python-string-formatting/problem?utm_source=chatgpt.com "String Formatting - HackerRank"
[2]: https://www.hackerrank.com/challenges/alphabet-rangoli/problem?utm_source=chatgpt.com "Alphabet Rangoli - HackerRank"
[3]: https://leetcode.com/problems/valid-number/?utm_source=chatgpt.com "65. Valid Number - LeetCode"
[4]: https://www.codewars.com/kata/57a386117cb1f31890000039/python?utm_source=chatgpt.com "Parse float | Codewars"
[5]: https://www.codewars.com/kata/557cd6882bfa3c8a9f0000c1?utm_source=chatgpt.com "Parse nice int from char problem - Codewars"
[6]: https://www.w3resource.com/python-exercises/generators-yield/python-generators-yield-exercise-14.php?utm_source=chatgpt.com "Python Armstrong number generator: Generate next Armstrong ... - w3resource"
[7]: https://zach.se/project-euler-solutions/6/?utm_source=chatgpt.com "Project Euler Problem 6 Solution - Zach Denton"
[8]: https://pynative.com/python-input-and-output-exercise/?utm_source=chatgpt.com "Python Input and Output Exercise with Solution [10 Exercise ... - PYnative"
[9]: https://stackoverflow.com/questions/73209895/how-to-read-all-inputs-from-stdin-in-hackerrank?utm_source=chatgpt.com "python - How to read all inputs from STDIN in HackerRank ... - Stack ..."
[10]: https://www.geeksforgeeks.org/python/python-least-frequent-character-in-string/?utm_source=chatgpt.com "Python - Least Frequent Character in String - GeeksforGeeks"
[11]: https://www.geeksforgeeks.org/python/sort-mixed-list-in-python/?utm_source=chatgpt.com "Sort mixed list in Python - GeeksforGeeks"
[12]: https://www.w3resource.com/python-exercises/basic/?utm_source=chatgpt.com "Python Basic (Part-II)- Exercises, Practice, Solution - w3resource"
[13]: https://pynative.com/python-read-specific-lines-from-a-file/?utm_source=chatgpt.com "Python Read Specific Lines From a File [5 Ways] – PYnative"
[14]: https://www.hackerrank.com/challenges/py-set-add/problem?utm_source=chatgpt.com "Set .add () - HackerRank"
[15]: https://www.hackerrank.com/challenges/symmetric-difference/problem?utm_source=chatgpt.com "Symmetric Difference - HackerRank"
[16]: https://www.geeksforgeeks.org/python/check-if-email-address-valid-or-not-in-python/?utm_source=chatgpt.com "Check if email address valid or not in Python - GeeksforGeeks"
[17]: https://realpython.com/command-line-interfaces-python-argparse/?utm_source=chatgpt.com "Build Command-Line Interfaces With Python's argparse"
[18]: https://www.tutorialspoint.com/How-to-Convert-Decimal-to-Binary-Using-Recursion-in-Python?utm_source=chatgpt.com "How to Convert Decimal to Binary Using Recursion in Python?"
[19]: https://www.programiz.com/python-programming/type-conversion-and-casting?utm_source=chatgpt.com "Python Type Conversion (With Examples) - Programiz"
[20]: https://docs.python.org/3/library/stdtypes.html?utm_source=chatgpt.com "Built-in Types — Python 3.13.5 documentation"


