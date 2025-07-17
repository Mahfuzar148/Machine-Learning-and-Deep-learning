
---

## 🛠️ 1. Python Setup – Jupyter / VS Code / Colab

* **Jupyter Notebook**: Ideal for data science & ML. Learn via DataQuest "Python functions and Jupyter" tutorial ([codechef.com][1], [Dataquest][2]).
* **VS Code Workflows**: Use the built‑in Jupyter integration. Microsoft video “Getting Started with Jupyter Notebooks in VS Code” shows how ([Microsoft Learn][3]).
* **Google Colab**: A free cloud notebook; begin with “Python-Basics.ipynb” demonstrating basics with live code ([Google Colab][4]).

#### ✅ Setup Tips:

```bash
# Create a virtual environment and install essentials
python -m venv env
source env/bin/activate        # On Windows: env\Scripts\activate
pip install numpy pandas jupyter matplotlib seaborn
```

---

## 📋 2. Variables, Data Types & I/O

* Python supports `int`, `float`, `str`, `bool`, `list`, `tuple`, `dict`, `set`.
* I/O basics:

  ```python
  name = input("Name? ")
  age = int(input("Age? "))
  print(f"Hello {name}, age {age}")
  ```
* Detailed guide available on GeeksforGeeks “Python programming language tutorial” ([GeeksforGeeks][5]).

---

## ➕➖ 3. Operators: Arithmetic, Logical & Comparison

* **Arithmetic**: `+ - * / // % ** @` (matrix mult) ([Wikipedia][6]).
* **Comparison**: `== != > >= < <=`
* **Logical**: `and`, `or`, `not`

**Example:**

```python
x, y = 5, 2
print(x ** y, x // y, x % y)
print(x > y and x < 10, not (x == y))
```

---

## 🔀 4. Conditional Statements

```python
n = int(input("Num: "))
if n % 2 == 0:
    print("Even")
elif n % 5 == 0:
    print("Multiple of 5")
else:
    print("Odd & not multiple of 5")
```

Practice more via GeeksforGeeks conditions & loops exercises ([Wikipedia][6], [GeeksforGeeks][7]).

---

## 🔄 5. Loops: `for`, `while`

```python
# for loop
for i in range(5):
    print(i * i)

# while loop
count = 0
while count < 5:
    print(count)
    count += 1
```

See practice sets at CodingBat (logic & loops) ([w3resource][8]).

---

## 🧩 6. Functions & Return Values

```python
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))

def factorial(n):
    if n < 0: raise ValueError("Non-negative only")
    res = 1
    for i in range(2, n+1):
        res *= i
    return res
print(factorial(5))
```

Integrated with Jupyter in DataQuest’s tutorial ([Dataquest][2]).

---

## 🧪 7. Curated Practice Resources

| Platform            | Focus                            | Problems/Topics Covered                                           |
| ------------------- | -------------------------------- | ----------------------------------------------------------------- |
| **Practice Python** | Beginner coding exercises        | 40+ tasks on lists, loops ([Practice Python][9], [CodingBat][10]) |
| **PYnative**        | Topic-specific drills (23 tasks) | Control flow, I/O, functions                                      |
| **GeeksforGeeks**   | Filtered Python problem bank     | Core basics + solutions                                           |
| **CodingBat**       | Logic & list/loop puzzles        | Practical for fundamentals                                        |

* For more advanced practice, explore **LeetCode (easy)** or **Exercism (Python track)** .

---

## 🧠 8. Additional Examples to Clarify

#### a) List comprehension + conditional:

```python
nums = [1,2,3,4,5,6]
evens = [x for x in nums if x % 2 == 0]
print(evens)  # [2,4,6]
```

#### b) Nested loops:

```python
for i in range(1,4):
    for j in range(1,4):
        print(f"{i} * {j} = {i*j}")
```

#### c) Recursion - Fibonacci:

```python
def fib(n):
    return n if n < 2 else fib(n-1) + fib(n-2)
print([fib(i) for i in range(10)])
```

---

## 📚 9. Suggested Readings & Tutorials

* **GeeksforGeeks Python tutorials** – in-depth with quizzes ([GeeksforGeeks][5])
* **DataQuest Jupyter + functions** – practical notebook guide ([Dataquest][2])
* **Colab basics notebook** – variables, loops inline ([Google Colab][4])
* **W3Schools exercises** – quick practice for each concept ([W3Schools][11])

---

## 🎯 Next Steps for Week 1

1. ✅ Set up your environment (locally or Colab).
2. Write small scripts covering I/O, loops, and functions.
3. Practice with at least 10 problems from the resources above.
4. Document all your attempts in a Jupyter notebook—using markdown & output cells.



[1]: https://www.codechef.com/practice/python?utm_source=chatgpt.com "Python Coding Practice Online: 195+ Problems on CodeChef"
[2]: https://www.dataquest.io/tutorial/python-functions-and-jupyter-notebook/?utm_source=chatgpt.com "Python Functions and Jupyter Notebook – Dataquest"
[3]: https://learn.microsoft.com/en-us/shows/visual-studio-code/getting-started-with-jupyter-notebooks-in-vs-code?utm_source=chatgpt.com "Getting Started with Jupyter Notebooks in VS Code"
[4]: https://colab.research.google.com/github/jckantor/CBE30338/blob/master/docs/01.02-Python-Basics.ipynb?utm_source=chatgpt.com "01.02-Python-Basics.ipynb - Colab"
[5]: https://www.geeksforgeeks.org/python/python-programming-language-tutorial/?utm_source=chatgpt.com "Python Tutorial - Learn Python Programming Language - GeeksforGeeks"
[6]: https://en.wikipedia.org/wiki/Python_%28programming_language%29?utm_source=chatgpt.com "Python (programming language)"
[7]: https://www.geeksforgeeks.org/python/python-exercises-practice-questions-and-solutions/?utm_source=chatgpt.com "Python Exercise with Practice Questions and Solutions"
[8]: https://www.w3resource.com/python-exercises/python-basic-exercises.php?utm_source=chatgpt.com "Python Basic: Exercises, Practice, Solution - w3resource"
[9]: https://www.practicepython.org/?utm_source=chatgpt.com "Practice Python"
[10]: https://codingbat.com/python?utm_source=chatgpt.com "CodingBat Python"
[11]: https://www.w3schools.com/python/python_exercises.asp?utm_source=chatgpt.com "Python Exercises - W3Schools"
