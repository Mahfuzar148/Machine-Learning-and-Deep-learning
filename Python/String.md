 Here is a **complete documentation on Python Strings** with syntax, examples, methods, and **practice problems with solutions**. This guide is excellent for both beginners and intermediate learners.

---

## 📘 Python String: Full Documentation

### 🧠 What is a String?

A **string** in Python is a **sequence of characters** enclosed in:

* single quotes `'Hello'`
* double quotes `"Hello"`
* triple quotes `'''Hello'''` (used for multi-line)

```python
s1 = 'Hello'
s2 = "World"
s3 = '''Multiline
String'''
```

---

### 📌 Creating Strings

```python
name = "David"
age = str(25)      # Convert int to string
```

---

### 📌 Accessing Characters & Slicing

```python
s = "Python"
print(s[0])      # P
print(s[-1])     # n
print(s[0:3])    # Pyt
print(s[::2])    # Pto
print(s[::-1])   # nohtyP
```

---

### 📌 String Immutability

Strings **cannot be changed** once created.

```python
s = "hello"
# s[0] = 'H'   ❌ Error: strings are immutable
s = 'H' + s[1:]  # ✅ Correct way
```

---

### 📌 Common String Operations

```python
a = "hello"
b = "world"
print(a + b)       # helloworld
print(a * 3)       # hellohellohello
print("e" in a)    # True
print(len(a))      # 5
```

---

### 📌 Useful Built-in String Methods

| Method              | Description                    | Example                                |
| ------------------- | ------------------------------ | -------------------------------------- |
| `lower()`           | Convert to lowercase           | `'Hello'.lower()` → `'hello'`          |
| `upper()`           | Convert to uppercase           | `'hi'.upper()` → `'HI'`                |
| `strip()`           | Remove leading/trailing spaces | `' hi '.strip()` → `'hi'`              |
| `replace(old, new)` | Replace substring              | `'cat'.replace('c', 'b')` → `'bat'`    |
| `split(delim)`      | Split string into list         | `'a,b,c'.split(',')` → `['a','b','c']` |
| `join(list)`        | Join list to string            | `','.join(['a','b'])` → `'a,b'`        |
| `find(sub)`         | Find index of substring        | `'hello'.find('l')` → `2`              |
| `count(sub)`        | Count occurrences              | `'banana'.count('a')` → `3`            |
| `startswith()`      | Check prefix                   | `'test.py'.startswith('test')`         |
| `endswith()`        | Check suffix                   | `'test.py'.endswith('.py')`            |
| `isalpha()`         | All letters?                   | `'abc'.isalpha()` → `True`             |
| `isdigit()`         | All digits?                    | `'123'.isdigit()` → `True`             |

---




Here is a **comprehensive list of Python string functions** with:

* ✅ **Function name**
* 💡 **What it does**
* 🔍 **Example with output**
* 📌 **Explanation**

---

## 🧵 Python String Functions – Full List with Examples

---

### 1. `str.upper()`

✅ Converts all characters to uppercase.

```python
"hello".upper()     # Output: 'HELLO'
```

📌 Converts each letter to its capital form.

---

### 2. `str.lower()`

✅ Converts all characters to lowercase.

```python
"HELLO".lower()     # Output: 'hello'
```

---

### 3. `str.title()`

✅ Capitalizes first letter of every word.

```python
"hello world".title()   # Output: 'Hello World'
```

---

### 4. `str.capitalize()`

✅ Capitalizes the first character only.

```python
"python rocks".capitalize()  # Output: 'Python rocks'
```

---

### 5. `str.swapcase()`

✅ Swaps case – upper becomes lower and vice versa.

```python
"PyThOn".swapcase()     # Output: 'pYtHoN'
```

---

### 6. `str.strip()`

✅ Removes leading and trailing spaces.

```python
"  hello  ".strip()     # Output: 'hello'
```

---

### 7. `str.lstrip()`

✅ Removes spaces from the **left** side only.

```python
"   hello".lstrip()     # Output: 'hello'
```

---

### 8. `str.rstrip()`

✅ Removes spaces from the **right** side only.

```python
"hello   ".rstrip()     # Output: 'hello'
```

---

### 9. `str.replace(old, new)`

✅ Replaces one substring with another.

```python
"banana".replace("a", "o")  # Output: 'bonono'
```

---

### 10. `str.split(sep)`

✅ Splits string into a list using separator.

```python
"red,blue,green".split(",")  # Output: ['red', 'blue', 'green']
```

---

### 11. `str.join(iterable)`

✅ Joins elements of a list using a string separator.

```python
"-".join(['a', 'b', 'c'])    # Output: 'a-b-c'
```

---

### 12. `str.find(sub)`

✅ Returns index of first occurrence of a substring.

```python
"hello".find("l")      # Output: 2
```

---

### 13. `str.rfind(sub)`

✅ Finds **last** occurrence of substring.

```python
"hello".rfind("l")     # Output: 3
```

---

### 14. `str.index(sub)`

✅ Like `find()`, but throws error if not found.

```python
"apple".index("p")     # Output: 1
```

---

### 15. `str.count(sub)`

✅ Counts how many times a substring occurs.

```python
"banana".count("a")    # Output: 3
```

---

### 16. `str.startswith(prefix)`

✅ Returns `True` if string starts with prefix.

```python
"hello.py".startswith("hello")  # Output: True
```

---

### 17. `str.endswith(suffix)`

✅ Returns `True` if string ends with suffix.

```python
"report.pdf".endswith(".pdf")  # Output: True
```

---

### 18. `str.isalpha()`

✅ Checks if all characters are letters.

```python
"abcXYZ".isalpha()     # Output: True
```

---

### 19. `str.isdigit()`

✅ Checks if all characters are digits.

```python
"12345".isdigit()      # Output: True
```

---

### 20. `str.isnumeric()`

✅ Like `isdigit()` but allows more digit types (like Unicode numerals).

```python
"Ⅻ".isnumeric()        # Output: True
```

---

### 21. `str.isalnum()`

✅ Checks if all characters are **alphanumeric** (letters or digits).

```python
"abc123".isalnum()     # Output: True
```

---

### 22. `str.isspace()`

✅ Checks if the string contains only whitespace.

```python
"   ".isspace()        # Output: True
```

---

### 23. `str.islower()`

✅ Checks if all letters are lowercase.

```python
"hello".islower()      # Output: True
```

---

### 24. `str.isupper()`

✅ Checks if all letters are uppercase.

```python
"HELLO".isupper()      # Output: True
```

---

### 25. `str.istitle()`

✅ Checks if string is in title case.

```python
"Hello World".istitle()    # Output: True
```

---

### 26. `str.zfill(width)`

✅ Pads string with leading zeros to match width.

```python
"42".zfill(5)          # Output: '00042'
```

---

### 27. `str.center(width, fillchar)`

✅ Centers the string with optional fill character.

```python
"hi".center(7, "*")    # Output: '**hi***'
```

---

### 28. `str.ljust(width, fillchar)`

✅ Left-align the string with padding.

```python
"hi".ljust(6, "_")     # Output: 'hi____'
```

---

### 29. `str.rjust(width, fillchar)`

✅ Right-align the string with padding.

```python
"hi".rjust(6, "_")     # Output: '____hi'
```

---

### 30. `str.partition(sep)`

✅ Splits string into 3 parts: before, sep, after.

```python
"email@example.com".partition("@")
# Output: ('email', '@', 'example.com')
```

---



### 📌 Formatting Strings

#### 1. f-strings (Python 3.6+)

```python
name = "David"
age = 25
print(f"My name is {name} and I am {age}")
```

#### 2. `format()` method

```python
print("My name is {} and I am {}".format(name, age))
```

#### 3. Old-style `%` formatting

```python
print("My name is %s and I am %d" % (name, age))
```

---

## 🧪 Practice Problems with Solutions

---

### 🔹 Problem 1: Reverse a String

```python
s = input("Enter a string: ")
print("Reversed:", s[::-1])
```

---

### 🔹 Problem 2: Count vowels in a string

```python
s = input("Enter string: ").lower()
vowels = 'aeiou'
count = sum(1 for c in s if c in vowels)
print("Vowel count:", count)
```

---

### 🔹 Problem 3: Check if string is palindrome

```python
s = input("Enter a string: ")
if s == s[::-1]:
    print("Palindrome")
else:
    print("Not palindrome")
```

---

### 🔹 Problem 4: Remove duplicates from string

```python
s = input("Enter a string: ")
result = ""
for ch in s:
    if ch not in result:
        result += ch
print("Without duplicates:", result)
```

---

### 🔹 Problem 5: Convert sentence to title case

```python
s = input("Enter sentence: ")
print("Title Case:", s.title())
```

---

### 🔹 Problem 6: Replace spaces with hyphens

```python
s = input("Enter string: ")
print(s.replace(" ", "-"))
```

---

### 🔹 Problem 7: Count each character in string

```python
from collections import Counter
s = input("Enter string: ")
counts = Counter(s)
for ch in sorted(counts):
    print(f"{ch}: {counts[ch]}")
```

---

### 🔹 Problem 8: Find longest word in sentence

```python
s = input("Enter sentence: ")
words = s.split()
longest = max(words, key=len)
print("Longest word:", longest)
```

---

### 🔹 Problem 9: Remove punctuation

```python
import string
s = input("Enter text: ")
cleaned = ''.join(ch for ch in s if ch not in string.punctuation)
print("Without punctuation:", cleaned)
```

---

### 🔹 Problem 10: Check anagram

```python
a = input("First string: ").replace(" ", "").lower()
b = input("Second string: ").replace(" ", "").lower()
print("Anagram" if sorted(a) == sorted(b) else "Not anagram")
```

---


### ✅ 11. **Count words in a string**

```python
s = input("Enter a sentence: ")
words = s.split()
print("Word count:", len(words))
```

> `split()` breaks the string into words by whitespace.

---

### ✅ 12. **Print characters at even index**

```python
s = input("Enter a string: ")
print("Characters at even indices:", s[::2])
```

> `s[::2]` slices every 2nd character starting from index 0.

---

### ✅ 13. **Print string in upper & lower**

```python
s = input("Enter a string: ")
print("Uppercase:", s.upper())
print("Lowercase:", s.lower())
```

> `upper()` and `lower()` are string methods.

---

### ✅ 14. **Check if string is numeric**

```python
s = input("Enter a string: ")
if s.isnumeric():
    print("It is numeric.")
else:
    print("It is not numeric.")
```

> `isnumeric()` returns True if all characters are digits.

---

### ✅ 15. **Find first repeating character**

```python
s = input("Enter a string: ")
seen = set()
for ch in s:
    if ch in seen:
        print("First repeating character:", ch)
        break
    seen.add(ch)
else:
    print("No repeating character found.")
```

> Uses a `set` to track previously seen characters.

---

### ✅ 16. **Print ASCII of each character**

```python
s = input("Enter a string: ")
for ch in s:
    print(f"{ch} → {ord(ch)}")
```

> `ord(ch)` returns the ASCII (Unicode) value of a character.

---

### ✅ 17. **Count capital & small letters**

```python
s = input("Enter a string: ")
upper = sum(1 for ch in s if ch.isupper())
lower = sum(1 for ch in s if ch.islower())
print("Uppercase letters:", upper)
print("Lowercase letters:", lower)
```

> Uses `isupper()` and `islower()` in generator expressions.

---

### ✅ 18. **Remove digits from string**

```python
s = input("Enter a string: ")
no_digits = ''.join(ch for ch in s if not ch.isdigit())
print("Without digits:", no_digits)
```

> `isdigit()` checks if character is a digit.

---

### ✅ 19. **Find common letters between two strings**

```python
a = input("Enter first string: ").lower()
b = input("Enter second string: ").lower()
common = set(a) & set(b)
print("Common letters:", ''.join(sorted(common)))
```

> Converts strings to sets to find intersection.

---

### ✅ 20. **Print word count using dictionary**

```python
s = input("Enter a sentence: ")
words = s.split()
word_count = {}
for word in words:
    word_count[word] = word_count.get(word, 0) + 1

for word, count in word_count.items():
    print(f"{word}: {count}")
```

> Uses `get()` to handle first occurrence of a word.

---



## 🧪 20 Practice Problems on Python String Functions

Each problem is **based on one or more string methods** (e.g. `upper()`, `split()`, `count()`, etc.).

---

### ✅ 1. Convert user input to UPPERCASE

```python
text = input("Enter a string: ")
print("Uppercase:", text.upper())
```

---

### ✅ 2. Convert string to lowercase and count vowels

```python
text = input("Enter a string: ").lower()
vowels = 'aeiou'
count = sum(1 for ch in text if ch in vowels)
print("Vowel count:", count)
```

---

### ✅ 3. Remove leading and trailing spaces

```python
s = input("Enter text with spaces: ")
print("Stripped text:", s.strip())
```

---

### ✅ 4. Replace all spaces with hyphens

```python
s = input("Enter a sentence: ")
print(s.replace(" ", "-"))
```

---

### ✅ 5. Count how many times “a” appears

```python
s = input("Enter a word: ")
print("Count of 'a':", s.count('a'))
```

---

### ✅ 6. Check if string starts with "https"

```python
url = input("Enter a URL: ")
print("Secure site?" , url.startswith("https"))
```

---

### ✅ 7. Find the position of first "@" in email

```python
email = input("Enter your email: ")
print("Position of '@':", email.find('@'))
```

---

### ✅ 8. Check if input is a number

```python
s = input("Enter something: ")
print("Is numeric?", s.isnumeric())
```

---

### ✅ 9. Check if string is alphabet-only

```python
name = input("Enter name: ")
print("Only letters?", name.isalpha())
```

---

### ✅ 10. Capitalize first letter only

```python
text = input("Enter a sentence: ")
print("Capitalized:", text.capitalize())
```

---

### ✅ 11. Convert “hello world” to Title Case

```python
text = input("Enter a sentence: ")
print("Title case:", text.title())
```

---

### ✅ 12. Count total words in a sentence

```python
text = input("Enter a sentence: ")
words = text.split()
print("Word count:", len(words))
```

---

### ✅ 13. Reverse the string

```python
s = input("Enter a string: ")
print("Reversed:", s[::-1])
```

---

### ✅ 14. Join a list of words with commas

```python
words = ['apple', 'banana', 'cherry']
print("CSV:", ",".join(words))
```

---

### ✅ 15. Replace all vowels with "\*"

```python
s = input("Enter a string: ")
for v in "aeiouAEIOU":
    s = s.replace(v, "*")
print(s)
```

---

### ✅ 16. Check if string contains only whitespace

```python
s = input("Enter text: ")
print("Only spaces?", s.isspace())
```

---

### ✅ 17. Convert number to string and pad with zeros

```python
n = input("Enter a number: ")
print("Zero-filled:", n.zfill(5))  # 0042 if input is 42
```

---

### ✅ 18. Count capital and lowercase letters

```python
s = input("Enter text: ")
upper = sum(1 for ch in s if ch.isupper())
lower = sum(1 for ch in s if ch.islower())
print("Uppercase:", upper)
print("Lowercase:", lower)
```

---

### ✅ 19. Split a CSV string and print values

```python
csv = input("Enter CSV (comma separated): ")
items = csv.split(",")
print("Items:", items)
```

---

### ✅ 20. Print ASCII value of each character

```python
s = input("Enter a word: ")
for ch in s:
    print(f"{ch}: {ord(ch)}")
```

---

## 🧾 Summary

This practice set covers:

* Case transformations
* Trimming, padding
* Counting, replacing, checking
* Splitting and joining
* Data validation (numeric, alphabet, etc.)

---


---

## 🧠 Codeforces Problems (Strings & I/O)

### 🔷 Beginner to Intermediate

| #  | Problem Name                                                           | Difficulty | Link                                  |
| -- | ---------------------------------------------------------------------- | ---------- | ------------------------------------- |
| 1  | [Petya and Strings](https://codeforces.com/problemset/problem/112/A)   | Easy       | Case-insensitive comparison           |
| 2  | [Boy or Girl](https://codeforces.com/problemset/problem/236/A)         | Easy       | Use of `set()` and string length      |
| 3  | [Word Capitalization](https://codeforces.com/problemset/problem/281/A) | Easy       | Use of `capitalize()`                 |
| 4  | [Way Too Long Words](https://codeforces.com/problemset/problem/71/A)   | Easy       | String length, abbreviation logic     |
| 5  | [Translation](https://codeforces.com/problemset/problem/41/A)          | Easy       | Reversal and comparison               |
| 6  | [Beautiful Year](https://codeforces.com/problemset/problem/271/A)      | Easy       | Type conversion, uniqueness of digits |
| 7  | [Dubstep](https://codeforces.com/problemset/problem/208/A)             | Easy       | `replace()` and `split()` logic       |
| 8  | [HQ9+](https://codeforces.com/problemset/problem/133/A)                | Easy       | Checking for certain characters       |
| 9  | [Palindrome](https://codeforces.com/problemset/problem/140A)           | Easy       | Reverse comparison                    |
| 10 | [Chat Room](https://codeforces.com/problemset/problem/58/A)            | Easy       | Substring pattern detection           |

---

### 🔷 Intermediate to Advanced

| #  | Problem Name                                                                  | Difficulty | Link                           |
| -- | ----------------------------------------------------------------------------- | ---------- | ------------------------------ |
| 11 | [Case of the Zeros and Ones](https://codeforces.com/problemset/problem/556/A) | Medium     | Removal logic                  |
| 12 | [String Task](https://codeforces.com/problemset/problem/118A)                 | Medium     | Vowel filtering and formatting |
| 13 | [Letters Rearrangement](https://codeforces.com/problemset/problem/978/B)      | Medium     | `count()` usage                |
| 14 | [Anagram](https://codeforces.com/problemset/problem/843/A)                    | Medium     | `Counter`, sorted comparison   |
| 15 | [Bear and Big Brother](https://codeforces.com/problemset/problem/791/A)       | Medium     | While loop with simple math    |

---

## 🧠 LeetCode Problems (Strings & Input/Output)

### 🔷 Easy Level

| # | Problem Name                                                                                                          | Topic Tags              | Link                    |
| - | --------------------------------------------------------------------------------------------------------------------- | ----------------------- | ----------------------- |
| 1 | [Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)                                                   | String, two pointers    | Remove non-alphanumeric |
| 2 | [Implement strStr()](https://leetcode.com/problems/implement-strstr/)                                                 | Substring search        | Basic `find()` or KMP   |
| 3 | [To Lower Case](https://leetcode.com/problems/to-lower-case/)                                                         | String conversion       | Use `.lower()`          |
| 4 | [Reverse String](https://leetcode.com/problems/reverse-string/)                                                       | Reverse logic           | Two pointers or slicing |
| 5 | [Reverse Words in a String III](https://leetcode.com/problems/reverse-words-in-a-string-iii/)                         | Word-level manipulation | `.split()` + slicing    |
| 6 | [Check If Two String Arrays are Equivalent](https://leetcode.com/problems/check-if-two-string-arrays-are-equivalent/) | Concatenation, equality | `join()` usage          |

---

### 🔷 Medium Level

| #  | Problem Name                                                                                                                    | Topic Tags               | Link                     |
| -- | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | ------------------------ |
| 7  | [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/) | Hashing, sliding window  | `set()` and `dict` usage |
| 8  | [Group Anagrams](https://leetcode.com/problems/group-anagrams/)                                                                 | String manipulation      | Use of sorted keys       |
| 9  | [Valid Anagram](https://leetcode.com/problems/valid-anagram/)                                                                   | Sorting, `Counter`       | Compare frequency        |
| 10 | [Multiply Strings](https://leetcode.com/problems/multiply-strings/)                                                             | Simulated multiplication | Manual digit operation   |
| 11 | [Integer to Roman](https://leetcode.com/problems/integer-to-roman/)                                                             | Mapping, loop            | Manual value-letter map  |
| 12 | [Roman to Integer](https://leetcode.com/problems/roman-to-integer/)                                                             | Mapping logic            | Use dict + loop          |

---

## 🧾 How to Use These for Learning

| Goal                                         | Suggested Platform                             |
| -------------------------------------------- | ---------------------------------------------- |
| 💻 **Fast typing and string I/O**            | Codeforces                                     |
| 🧠 **Data type handling + string logic**     | LeetCode                                       |
| 🔄 **Practice conversion and formatting**    | LeetCode easy/medium                           |
| 🔍 **Master string patterns and conditions** | Codeforces problems on substrings and reversal |

---




