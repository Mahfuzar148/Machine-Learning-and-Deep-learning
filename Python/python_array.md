

## ✅ Problem 1: Create a list of 10 even numbers

### 🧠 Description:

Generate a list of the first 10 even numbers starting from 2.

### 📝 Explanation:

Even numbers are those divisible by 2. To get the first 10 even numbers, we can use `range(start, stop, step)` with step 2 starting from 2.

### 💡 Code:

```python
even_numbers = [i for i in range(2, 21, 2)]
print(even_numbers)
```

---

## ✅ Problem 2: Find the maximum value in a list

### 🧠 Description:

Find the largest number from a given list.

### 📝 Explanation:

Python's built-in `max()` function gives you the highest value in a list instantly.

### 💡 Code:

```python
numbers = [4, 10, 15, 2, 9]
max_value = max(numbers)
print("Maximum value:", max_value)
```

---

## ✅ Problem 3: Find the index of a specific element in an array

### 🧠 Description:

Find where a specific element is located in a list.

### 📝 Explanation:

The `.index()` method returns the position (index) of the first occurrence of a value.

### 💡 Code:

```python
arr = [5, 3, 7, 1, 9]
index = arr.index(7)
print("Index of 7:", index)
```

---

## ✅ Problem 4: Reverse an array without using reverse()

### 🧠 Description:

Print the elements of a list in reverse order, without using the `.reverse()` method.

### 📝 Explanation:

List slicing with `[::-1]` gives the reversed version of a list.

### 💡 Code:

```python
arr = [1, 2, 3, 4, 5]
reversed_arr = arr[::-1]
print("Reversed:", reversed_arr)
```

---

## ✅ Problem 5: Count the number of zeros in a list

### 🧠 Description:

Count how many times `0` appears in a list.

### 📝 Explanation:

Use the `.count()` method to count the frequency of a specific value in a list.

### 💡 Code:

```python
arr = [0, 1, 2, 0, 4, 0]
zero_count = arr.count(0)
print("Zeros:", zero_count)
```

---


---

## ✅ Problem 6: Remove duplicates from an array

### 🧠 Description:

Remove all duplicate elements from a list, keeping only unique ones.

### 📝 Explanation:

Python’s `set()` removes duplicates. Convert list → set → list again to retain unique items.

### 💡 Code:

```python
arr = [1, 2, 2, 3, 4, 4, 5]
unique = list(set(arr))
print("Unique elements:", unique)
```

---

## ✅ Problem 7: Sum all even numbers in a list

### 🧠 Description:

Add up all even numbers present in a given list.

### 📝 Explanation:

Loop through the list and add values that are divisible by 2 (`num % 2 == 0`).

### 💡 Code:

```python
arr = [1, 2, 3, 4, 5, 6]
total = sum(num for num in arr if num % 2 == 0)
print("Sum of even numbers:", total)
```

---

## ✅ Problem 8: Find the second largest element in a list

### 🧠 Description:

Find the second highest number in a list.

### 📝 Explanation:

Convert to set to remove duplicates → sort → pick second last item.

### 💡 Code:

```python
arr = [10, 20, 4, 45, 99]
unique_sorted = sorted(set(arr))
second_largest = unique_sorted[-2]
print("Second largest:", second_largest)
```

---

## ✅ Problem 9: Check if a list is a palindrome

### 🧠 Description:

Check whether the list reads the same forwards and backwards.

### 📝 Explanation:

Compare the list with its reversed version using slicing.

### 💡 Code:

```python
arr = [1, 2, 3, 2, 1]
is_palindrome = arr == arr[::-1]
print("Is palindrome?", is_palindrome)
```

---

## ✅ Problem 10: Sort a list without using sort()

### 🧠 Description:

Sort a list manually (without using `sort()` or `sorted()`).

### 📝 Explanation:

Use a sorting algorithm like **Bubble Sort** for educational purposes.

### 💡 Code:

```python
arr = [5, 2, 9, 1, 5, 6]
for i in range(len(arr)):
    for j in range(i + 1, len(arr)):
        if arr[i] > arr[j]:
            arr[i], arr[j] = arr[j], arr[i]
print("Sorted array:", arr)
```

---

## ✅ Problem 11: Merge two sorted arrays into one

### 🧠 Description:

Combine two already sorted lists into one sorted list.

### 📝 Explanation:

Use two pointers to merge like in Merge Sort.

### 💡 Code:

```python
a = [1, 3, 5]
b = [2, 4, 6]
merged = sorted(a + b)
print("Merged:", merged)
```

---

## ✅ Problem 12: Rotate an array to the right by k steps

### 🧠 Description:

Move each element `k` positions to the right.

### 📝 Explanation:

Use slicing to rotate efficiently: `arr[-k:] + arr[:-k]`

### 💡 Code:

```python
arr = [1, 2, 3, 4, 5]
k = 2
rotated = arr[-k:] + arr[:-k]
print("Rotated array:", rotated)
```

---

## ✅ Problem 13: Find common elements between two lists

### 🧠 Description:

Identify values that appear in both lists.

### 📝 Explanation:

Use `set` intersection to find common elements.

### 💡 Code:

```python
a = [1, 2, 3, 4]
b = [3, 4, 5, 6]
common = list(set(a) & set(b))
print("Common elements:", common)
```

---

## ✅ Problem 14: Separate even and odd numbers from a list

### 🧠 Description:

Split a list into two lists: one for even numbers, one for odd.

### 📝 Explanation:

Use list comprehension with conditions.

### 💡 Code:

```python
arr = [1, 2, 3, 4, 5, 6]
even = [x for x in arr if x % 2 == 0]
odd = [x for x in arr if x % 2 != 0]
print("Even:", even)
print("Odd:", odd)
```

---

## ✅ Problem 15: Replace all negative numbers with 0

### 🧠 Description:

Turn all negative numbers in the list into 0.

### 📝 Explanation:

Use list comprehension with a condition.

### 💡 Code:

```python
arr = [1, -2, 3, -4, 5]
non_negative = [x if x >= 0 else 0 for x in arr]
print("No negatives:", non_negative)
```

---

## ✅ Problem 16: Flatten a 2D array into 1D

### 🧠 Description:

Convert a 2D list into a single 1D list.

### 📝 Explanation:

Use nested loops or list comprehension.

### 💡 Code:

```python
arr = [[1, 2], [3, 4], [5]]
flat = [item for sublist in arr for item in sublist]
print("Flattened:", flat)
```

---

## ✅ Problem 17: Find the frequency of each element in a list

### 🧠 Description:

Count how many times each value appears in a list.

### 📝 Explanation:

Use a dictionary or `collections.Counter`.

### 💡 Code:

```python
from collections import Counter
arr = [1, 2, 2, 3, 3, 3]
freq = Counter(arr)
print("Frequencies:", freq)
```

---

## ✅ Problem 18: Split a list into chunks of size n

### 🧠 Description:

Break a list into smaller lists (chunks) of given size.

### 📝 Explanation:

Use slicing inside a loop.

### 💡 Code:

```python
arr = [1, 2, 3, 4, 5, 6, 7]
n = 3
chunks = [arr[i:i+n] for i in range(0, len(arr), n)]
print("Chunks:", chunks)
```

---

## ✅ Problem 19: Check if two arrays are equal

### 🧠 Description:

Determine if two lists contain the same elements in the same order.

### 📝 Explanation:

Just compare directly with `==`.

### 💡 Code:

```python
a = [1, 2, 3]
b = [1, 2, 3]
print("Equal?" , a == b)
```

---

## ✅ Problem 20: Find the median of a list

### 🧠 Description:

Find the middle value in a sorted list.

### 📝 Explanation:

Sort the list. If odd number of elements, return the middle. If even, average the two middle values.

### 💡 Code:

```python
arr = [5, 3, 1, 4, 2]
arr.sort()
n = len(arr)
if n % 2 == 1:
    median = arr[n//2]
else:
    median = (arr[n//2 - 1] + arr[n//2]) / 2
print("Median:", median)
```

---


---

## ✅ Problem 21: Find the mode of a list

### 🧠 Description:

Find the most frequent element(s) in a list.

### 📝 Explanation:

Use `collections.Counter` to count frequency, then extract the element(s) with max count.

### 💡 Code:

```python
from collections import Counter
arr = [1, 2, 2, 3, 3, 3]
freq = Counter(arr)
mode = [k for k, v in freq.items() if v == max(freq.values())]
print("Mode:", mode)
```

---

## ✅ Problem 22: Remove all occurrences of a given element from a list

### 🧠 Description:

Remove every occurrence of a specific number.

### 📝 Explanation:

Use list comprehension to keep elements that are not equal to the target.

### 💡 Code:

```python
arr = [1, 2, 3, 2, 4, 2]
target = 2
filtered = [x for x in arr if x != target]
print("After removal:", filtered)
```

---

## ✅ Problem 23: Count the number of prime numbers in a list

### 🧠 Description:

Count how many elements are prime numbers.

### 📝 Explanation:

Write a helper function `is_prime()`, then count the primes using list comprehension.

### 💡 Code:

```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True

arr = [2, 3, 4, 5, 6, 7, 8]
prime_count = sum(1 for x in arr if is_prime(x))
print("Prime count:", prime_count)
```

---

## ✅ Problem 24: Insert an element at a specific position in a list

### 🧠 Description:

Insert a value at a given index.

### 📝 Explanation:

Use `.insert(index, value)`.

### 💡 Code:

```python
arr = [1, 2, 3]
arr.insert(1, 99)
print("After insert:", arr)
```

---

## ✅ Problem 25: Remove an element from a specific index

### 🧠 Description:

Delete the item at a particular position.

### 📝 Explanation:

Use `del arr[index]` or `pop(index)`.

### 💡 Code:

```python
arr = [1, 2, 3, 4]
del arr[2]
print("After removal:", arr)
```

---

## ✅ Problem 26: Create a list of squares of numbers from 1 to 10

### 🧠 Description:

Build a list of square numbers.

### 📝 Explanation:

Use list comprehension: `x**2`.

### 💡 Code:

```python
squares = [x**2 for x in range(1, 11)]
print("Squares:", squares)
```

---

## ✅ Problem 27: Print all subarrays of a given array

### 🧠 Description:

Print all possible contiguous subarrays.

### 📝 Explanation:

Use nested loops: start and end index combinations.

### 💡 Code:

```python
arr = [1, 2, 3]
for i in range(len(arr)):
    for j in range(i+1, len(arr)+1):
        print(arr[i:j])
```

---

## ✅ Problem 28: Find the longest increasing subsequence in a list

### 🧠 Description:

Find the longest strictly increasing sequence (not necessarily contiguous).

### 📝 Explanation:

Use dynamic programming (basic version).

### 💡 Code:

def LIS(arr):
n = len(arr)
dp = \[1]\*n
for i in range(n):
for j in range(i):
if arr\[i] > arr\[j]:
dp\[i] = max(dp\[i], dp\[j]+1)
return max(dp)

arr = \[10, 22, 9, 33, 21, 50]
print("Length of LIS:", LIS(arr))

---

## ✅ Problem 29: Check if a list contains only unique elements

### 🧠 Description:

Detect duplicates.

### 📝 Explanation:

Compare `len(arr)` with `len(set(arr))`.

### 💡 Code:

```python
arr = [1, 2, 3, 2]
print("Is unique?", len(arr) == len(set(arr)))
```

---

## ✅ Problem 30: Find the first non-repeating element in a list

### 🧠 Description:

Get the first value that appears only once.

### 📝 Explanation:

Use `collections.Counter` to count elements, loop to find first with count = 1.

### 💡 Code:

```python
from collections import Counter
arr = [9, 4, 9, 6, 7, 4]
freq = Counter(arr)
for x in arr:
    if freq[x] == 1:
        print("First non-repeating:", x)
        break
```

---

## ✅ Problem 31: Swap the first and last element in a list

### 🧠 Description:

Switch the first and last items.

### 📝 Explanation:

Use simultaneous assignment.

### 💡 Code:

```python
arr = [1, 2, 3, 4]
arr[0], arr[-1] = arr[-1], arr[0]
print("Swapped list:", arr)
```

---

## ✅ Problem 32: Find the average of elements in a list

### 🧠 Description:

Calculate mean value.

### 📝 Explanation:

Sum all and divide by length.

### 💡 Code:

```python
arr = [10, 20, 30]
avg = sum(arr) / len(arr)
print("Average:", avg)
```

---

## ✅ Problem 33: Remove all elements greater than a given number

### 🧠 Description:

Filter list by max threshold.

### 📝 Explanation:

Use list comprehension with condition.

### 💡 Code:

```python
arr = [10, 25, 30, 5]
threshold = 20
filtered = [x for x in arr if x <= threshold]
print("Filtered:", filtered)
```

---

## ✅ Problem 34: Check if all elements in a list are numbers

### 🧠 Description:

Verify all values are `int` or `float`.

### 📝 Explanation:

Use `isinstance()` in a loop or all().

### 💡 Code:

```python
arr = [1, 2.5, 3]
print("All numbers?", all(isinstance(x, (int, float)) for x in arr))
```

---

## ✅ Problem 35: Print the multiplication table of a number

### 🧠 Description:

Show multiplication from 1 to 10.

### 📝 Explanation:

Use a simple `for` loop with `range`.

### 💡 Code:

```python
n = 5
for i in range(1, 11):
    print(f"{n} x {i} = {n*i}")
```

---

## ✅ Problem 36: Print all prime numbers from 1 to 100

### 🧠 Description:

Display primes between 1 and 100.

### 📝 Explanation:

Use `is_prime()` function inside a loop.

### 💡 Code:

```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True

for i in range(1, 101):
    if is_prime(i):
        print(i, end=' ')
```

---

## ✅ Problem 37: Calculate factorial using a for loop

### 🧠 Description:

Multiply all numbers from 1 to n.

### 📝 Explanation:

Use loop and accumulator variable.

### 💡 Code:

```python
n = 5
fact = 1
for i in range(1, n+1):
    fact *= i
print("Factorial:", fact)
```

---

## ✅ Problem 38: Find the sum of digits of a number

### 🧠 Description:

Add each digit separately.

### 📝 Explanation:

Convert number to string or use math operations.

### 💡 Code:

```python
n = 1234
digit_sum = sum(int(d) for d in str(n))
print("Digit sum:", digit_sum)
```

---

## ✅ Problem 39: Print Fibonacci series up to n terms

### 🧠 Description:

Generate first n terms of Fibonacci.

### 📝 Explanation:

Use a loop and track two variables.

### 💡 Code:

```python
n = 10
a, b = 0, 1
for _ in range(n):
    print(a, end=' ')
    a, b = b, a + b
```

---

## ✅ Problem 40: Count vowels in a string

### 🧠 Description:

Count how many vowels (a, e, i, o, u) are in a string.

### 📝 Explanation:

Loop through characters and check if in vowel set.

### 💡 Code:

```python
s = "Hello World"
vowels = "aeiouAEIOU"
count = sum(1 for c in s if c in vowels)
print("Vowel count:", count)
```

---



---

## ✅ Problem 41: Count uppercase letters in a string

### 🧠 Description:

Find how many characters in a string are uppercase letters (A–Z).

### 📝 Explanation:

Use a loop or comprehension and check with `.isupper()`.

### 💡 Code:

```python
s = "Hello World"
uppercase_count = sum(1 for c in s if c.isupper())
print("Uppercase letters:", uppercase_count)
```

---

## ✅ Problem 42: Print a triangle pattern using stars

### 🧠 Description:

Create a triangle using `*` symbols.

### 📝 Explanation:

Use a loop to print increasing number of stars per line.

### 💡 Code:

```python
n = 5
for i in range(1, n+1):
    print("*" * i)
```

---

## ✅ Problem 43: Generate a list of factorials from 1 to n

### 🧠 Description:

Create a list where each element is the factorial of its index from 1 to n.

### 📝 Explanation:

Use a loop and multiply progressively.

### 💡 Code:

```python
n = 5
fact = 1
factorials = []
for i in range(1, n+1):
    fact *= i
    factorials.append(fact)
print("Factorials:", factorials)
```

---

## ✅ Problem 44: Print numbers divisible by 3 and 5 from 1 to 100

### 🧠 Description:

Filter numbers divisible by both 3 and 5 in a given range.

### 📝 Explanation:

Use `if i % 3 == 0 and i % 5 == 0`.

### 💡 Code:

```python
for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print(i, end=' ')
```

---

## ✅ Problem 45: Calculate the power of a number using loops

### 🧠 Description:

Compute `base^exponent` using repeated multiplication.

### 📝 Explanation:

Multiply `base` by itself `exp` times in a loop.

### 💡 Code:

```python
base = 2
exp = 5
result = 1
for _ in range(exp):
    result *= base
print("Power:", result)
```

---

## ✅ Problem 46: Check if a string is a palindrome using a loop

### 🧠 Description:

Manually verify if a string reads the same backward.

### 📝 Explanation:

Compare characters from both ends.

### 💡 Code:

```python
s = "madam"
is_palindrome = True
for i in range(len(s) // 2):
    if s[i] != s[-i - 1]:
        is_palindrome = False
        break
print("Is palindrome?", is_palindrome)
```

---

## ✅ Problem 47: Print ASCII values of characters from A to Z

### 🧠 Description:

Display the ASCII values of capital letters.

### 📝 Explanation:

Use `ord()` to get ASCII codes.

### 💡 Code:

```python
for c in range(ord('A'), ord('Z')+1):
    print(f"{chr(c)} : {c}")
```

---

## ✅ Problem 48: Count the number of words in a sentence

### 🧠 Description:

Determine how many words are in a string.

### 📝 Explanation:

Split the sentence using `.split()` and count parts.

### 💡 Code:

```python
s = "This is a sample sentence"
word_count = len(s.split())
print("Word count:", word_count)
```

---

## ✅ Problem 49: Print the reverse of a string using a loop

### 🧠 Description:

Reverse a string manually without using slicing.

### 📝 Explanation:

Use a loop and concatenate characters in reverse order.

### 💡 Code:

```python
s = "hello"
rev = ""
for c in s:
    rev = c + rev
print("Reversed:", rev)
```

---

## ✅ Problem 50: Print elements of a matrix using nested loops

### 🧠 Description:

Print each element from a 2D list.

### 📝 Explanation:

Use one loop for rows and one for columns.

### 💡 Code:

```python
matrix = [[1, 2], [3, 4], [5, 6]]
for row in matrix:
    for item in row:
        print(item, end=' ')
    print()
```

---

## ✅ Problem 51: Find the largest number in a nested list

### 🧠 Description:

Get the maximum value from a list of lists.

### 📝 Explanation:

Flatten and use `max()`.

### 💡 Code:

```python
nested = [[1, 3], [7, 2], [4, 9]]
flat = [num for sublist in nested for num in sublist]
print("Max:", max(flat))
```

---

## ✅ Problem 52: Check if a number is an Armstrong number

### 🧠 Description:

Check if sum of cubes of digits equals the number.

### 📝 Explanation:

Break digits, cube each, compare with original.

### 💡 Code:

```python
n = 153
digits = [int(d) for d in str(n)]
is_armstrong = sum(d**3 for d in digits) == n
print("Is Armstrong?", is_armstrong)
```

---

## ✅ Problem 53: Find GCD of two numbers using loops

### 🧠 Description:

Find the greatest common divisor of two numbers.

### 📝 Explanation:

Loop from 1 to min(a, b) and check divisibility.

### 💡 Code:

```python
a, b = 60, 48
gcd = 1
for i in range(1, min(a, b)+1):
    if a % i == 0 and b % i == 0:
        gcd = i
print("GCD:", gcd)
```

---

## ✅ Problem 54: Print a diamond pattern using stars

### 🧠 Description:

Print a symmetrical diamond using `*`.

### 📝 Explanation:

Use nested loops to align stars.

### 💡 Code:

```python
n = 5
for i in range(n):
    print(" "*(n-i-1) + "*"*(2*i+1))
for i in range(n-2, -1, -1):
    print(" "*(n-i-1) + "*"*(2*i+1))
```

---

## ✅ Problem 55: Sum of even digits of a number

### 🧠 Description:

Add all even digits in an integer.

### 📝 Explanation:

Convert to string, check digits.

### 💡 Code:

```python
n = 123456
even_sum = sum(int(d) for d in str(n) if int(d) % 2 == 0)
print("Sum of even digits:", even_sum)
```

---

## ✅ Problem 56: Convert a string to uppercase without using built-in functions

### 🧠 Description:

Transform lowercase to uppercase using ASCII.

### 📝 Explanation:

Subtract 32 from ASCII value of lowercase letters.

### 💡 Code:

```python
s = "hello"
upper = ""
for c in s:
    if 'a' <= c <= 'z':
        upper += chr(ord(c) - 32)
    else:
        upper += c
print("Uppercase:", upper)
```

---

## ✅ Problem 57: Generate a list of cubes from 1 to 10

### 🧠 Description:

Create a list of cube numbers.

### 📝 Explanation:

Use list comprehension.

### 💡 Code:

```python
cubes = [x**3 for x in range(1, 11)]
print("Cubes:", cubes)
```

---

## ✅ Problem 58: Print a number pyramid pattern

### 🧠 Description:

Display numbers in pyramid form.

### 📝 Explanation:

Use nested loops and formatting.

### 💡 Code:

```python
n = 5
for i in range(1, n+1):
    print(" "*(n-i) + " ".join(str(j) for j in range(1, i+1)))
```

---

## ✅ Problem 59: Calculate the sum of squares of first n natural numbers

### 🧠 Description:

Find 1² + 2² + ... + n²

### 📝 Explanation:

Use a loop or formula `n(n+1)(2n+1)/6`.

### 💡 Code:

```python
n = 5
sum_squares = sum(i**2 for i in range(1, n+1))
print("Sum of squares:", sum_squares)
```

---

## ✅ Problem 60: Print the count of digits in a number

### 🧠 Description:

Count how many digits are in a number.

### 📝 Explanation:

Use `len(str(n))`.

### 💡 Code:

```python
n = 123456
print("Digit count:", len(str(n)))
```

---

---

## ✅ Problem 61: Count the frequency of characters in a string

### 🧠 Description:

Determine how often each character appears in a string.

### 📝 Explanation:

Use `collections.Counter` to tally character counts.

### 💡 Code:

```python
from collections import Counter
s = "programming"
freq = Counter(s)
print("Character frequencies:", freq)
```

---

## ✅ Problem 62: Find common characters between two strings

### 🧠 Description:

Identify letters that appear in both strings.

### 📝 Explanation:

Use `set` intersection.

### 💡 Code:

```python
a = "python"
b = "typhoon"
common = set(a) & set(b)
print("Common characters:", common)
```

---

## ✅ Problem 63: Print the calendar of a given month and year

### 🧠 Description:

Show the calendar of a specific month.

### 📝 Explanation:

Use the `calendar` module.

### 💡 Code:

```python
import calendar
year = 2025
month = 7
print(calendar.month(year, month))
```

---

## ✅ Problem 64: Check if a list is sorted in ascending order

### 🧠 Description:

Determine whether a list is already sorted.

### 📝 Explanation:

Compare list with its sorted version.

### 💡 Code:

```python
arr = [1, 2, 3, 4]
is_sorted = arr == sorted(arr)
print("Is sorted?", is_sorted)
```

---

## ✅ Problem 65: Find sum of all odd numbers in a list

### 🧠 Description:

Add only the odd elements in a list.

### 📝 Explanation:

Check `num % 2 != 0` in loop.

### 💡 Code:

```python
arr = [1, 2, 3, 4, 5]
odd_sum = sum(x for x in arr if x % 2 != 0)
print("Sum of odd numbers:", odd_sum)
```

---

## ✅ Problem 66: Create a multiplication table for numbers 1 to 10

### 🧠 Description:

Generate full multiplication table from 1 to 10.

### 📝 Explanation:

Use nested loops for rows and columns.

### 💡 Code:

```python
for i in range(1, 11):
    for j in range(1, 11):
        print(f"{i*j:4}", end="")
    print()
```

---

## ✅ Problem 67: Print all even indexed elements of a list

### 🧠 Description:

Display only elements at even index positions.

### 📝 Explanation:

Use slicing: `arr[::2]`.

### 💡 Code:

```python
arr = [10, 20, 30, 40, 50]
print("Even index elements:", arr[::2])
```

---

## ✅ Problem 68: Print numbers from 1 to n using a while loop

### 🧠 Description:

Use `while` to count from 1 to n.

### 📝 Explanation:

Start from 1, increment until `n`.

### 💡 Code:

```python
n = 5
i = 1
while i <= n:
    print(i, end=' ')
    i += 1
```

---

## ✅ Problem 69: Calculate the factorial of a number using while loop

### 🧠 Description:

Multiply from `n` down to 1.

### 📝 Explanation:

Use decrementing `while` loop.

### 💡 Code:

```python
n = 5
fact = 1
while n > 1:
    fact *= n
    n -= 1
print("Factorial:", fact)
```

---

## ✅ Problem 70: Print Fibonacci series using while loop

### 🧠 Description:

Generate Fibonacci numbers with while loop.

### 📝 Explanation:

Keep adding previous two numbers.

### 💡 Code:

```python
n = 10
a, b = 0, 1
count = 0
while count < n:
    print(a, end=' ')
    a, b = b, a + b
    count += 1
```

---

## ✅ Problem 71: Count number of digits in a number

### 🧠 Description:

Figure out how many digits are in a number.

### 📝 Explanation:

Convert to string or divide repeatedly.

### 💡 Code:

```python
n = 123456
count = 0
while n > 0:
    n //= 10
    count += 1
print("Digit count:", count)
```

---

## ✅ Problem 72: Reverse a number using while loop

### 🧠 Description:

Flip the digits of a number.

### 📝 Explanation:

Use mod and divide.

### 💡 Code:

```python
n = 1234
rev = 0
while n > 0:
    rev = rev * 10 + n % 10
    n //= 10
print("Reversed:", rev)
```

---

## ✅ Problem 73: Check if a number is palindrome using while

### 🧠 Description:

Same digits forward and backward.

### 📝 Explanation:

Reverse number, compare to original.

### 💡 Code:

```python
n = 121
original = n
rev = 0
while n > 0:
    rev = rev * 10 + n % 10
    n //= 10
print("Is palindrome?", rev == original)
```

---

## ✅ Problem 74: Sum digits until it becomes a single digit

### 🧠 Description:

Repeat sum of digits until one digit remains.

### 📝 Explanation:

Use a loop that checks if number > 9.

### 💡 Code:

```python
n = 9875
while n > 9:
    n = sum(int(d) for d in str(n))
print("Single digit sum:", n)
```

---

## ✅ Problem 75: Keep taking input until user types 'exit'

### 🧠 Description:

Take input continuously until user says "exit".

### 📝 Explanation:

Use infinite loop with break condition.

### 💡 Code:

```python
while True:
    inp = input("Enter something: ")
    if inp.lower() == 'exit':
        break
    print("You typed:", inp)
```

---

## ✅ Problem 76: Print multiplication table using while loop

### 🧠 Description:

Create a table with while instead of for.

### 📝 Explanation:

Use counter and multiply.

### 💡 Code:

```python
n = 5
i = 1
while i <= 10:
    print(f"{n} x {i} = {n*i}")
    i += 1
```

---

## ✅ Problem 77: Generate a countdown from n to 1

### 🧠 Description:

Print reverse numbers using while.

### 📝 Explanation:

Start at `n`, go down to 1.

### 💡 Code:

```python
n = 10
while n >= 1:
    print(n, end=' ')
    n -= 1
```

---

## ✅ Problem 78: Calculate power using while loop

### 🧠 Description:

Raise base to exponent.

### 📝 Explanation:

Multiply base repeatedly.

### 💡 Code:

```python
base = 2
exp = 4
result = 1
while exp > 0:
    result *= base
    exp -= 1
print("Result:", result)
```

---

## ✅ Problem 79: Keep dividing a number by 2 until it becomes 1

### 🧠 Description:

Halve repeatedly until 1.

### 📝 Explanation:

Use loop with `n //= 2`.

### 💡 Code:

```python
n = 64
while n > 1:
    print(n, end=' ')
    n //= 2
```

---

## ✅ Problem 80: Sum of first n even numbers using while

### 🧠 Description:

Add first `n` even numbers starting from 2.

### 📝 Explanation:

Use counter and accumulator.

### 💡 Code:

```python
n = 5
i = 1
count = 0
total = 0
while count < n:
    if i % 2 == 0:
        total += i
        count += 1
    i += 1
print("Sum of even numbers:", total)
```

---

---

## ✅ Problem 81: Print the square of numbers from 1 to n

### 🧠 Description:

Print each number's square from 1 to `n`.

### 💡 Code:

```python
n = 5
i = 1
while i <= n:
    print(f"{i}^2 =", i*i)
    i += 1
```

---

## ✅ Problem 82: Sum the digits of a number using while loop

### 🧠 Description:

Add all digits of an integer using a while loop.

### 💡 Code:

```python
n = 1234
total = 0
while n > 0:
    total += n % 10
    n //= 10
print("Sum of digits:", total)
```

---

## ✅ Problem 83: Keep printing characters until a vowel is found

### 🧠 Description:

Loop through string and stop when you find a vowel.

### 💡 Code:

```python
s = "bcdfghello"
i = 0
while i < len(s):
    if s[i].lower() in 'aeiou':
        print("Vowel found:", s[i])
        break
    print(s[i])
    i += 1
```

---

## ✅ Problem 84: Simulate a login system with 3 tries using while

### 🧠 Description:

Ask user for password max 3 times.

### 💡 Code:

```python
correct_password = "admin123"
attempts = 3

while attempts > 0:
    pw = input("Enter password: ")
    if pw == correct_password:
        print("Access granted.")
        break
    else:
        attempts -= 1
        print(f"Wrong password. {attempts} tries left.")
else:
    print("Account locked.")
```

---

## ✅ Problem 85: Generate a number guessing game

### 🧠 Description:

User keeps guessing until they find the number.

### 💡 Code:

```python
import random
target = random.randint(1, 10)
guess = None

while guess != target:
    guess = int(input("Guess a number (1–10): "))
    if guess < target:
        print("Too low!")
    elif guess > target:
        print("Too high!")
print("Correct!")
```

---

## ✅ Problem 86: Keep asking for numbers and print their sum until 0 is entered

### 🧠 Description:

Continue taking input and summing numbers until user enters 0.

### 💡 Code:

```python
total = 0
while True:
    num = int(input("Enter a number (0 to stop): "))
    if num == 0:
        break
    total += num
print("Total sum:", total)
```

---

## ✅ Problem 87: Print only positive numbers from a list using while

### 🧠 Description:

Iterate a list and print only positive numbers.

### 💡 Code:

```python
arr = [-2, 3, -1, 5, 0]
i = 0
while i < len(arr):
    if arr[i] > 0:
        print(arr[i])
    i += 1
```

---

## ✅ Problem 88: Keep appending user inputs to a list until length is 5

### 🧠 Description:

Create a list of 5 items by asking input from user.

### 💡 Code:

```python
items = []
while len(items) < 5:
    items.append(input("Enter item: "))
print("Final list:", items)
```

---

## ✅ Problem 89: Find smallest divisor of a number using while

### 🧠 Description:

Find the smallest number that divides `n` other than 1.

### 💡 Code:

```python
n = 49
div = 2
while n % div != 0:
    div += 1
print("Smallest divisor:", div)
```

---

## ✅ Problem 90: Find LCM of two numbers using while loop

### 🧠 Description:

Least common multiple using increment method.

### 💡 Code:

```python
a, b = 6, 8
lcm = max(a, b)
while True:
    if lcm % a == 0 and lcm % b == 0:
        break
    lcm += 1
print("LCM:", lcm)
```

---

## ✅ Problem 91: Print elements of a list in reverse using while

### 🧠 Description:

Use `while` to go backward through a list.

### 💡 Code:

```python
arr = [10, 20, 30, 40]
i = len(arr) - 1
while i >= 0:
    print(arr[i])
    i -= 1
```

---

## ✅ Problem 92: Count spaces in a string using while

### 🧠 Description:

Count how many spaces `' '` are in a string.

### 💡 Code:

```python
s = "a b c d e"
i = 0
spaces = 0
while i < len(s):
    if s[i] == ' ':
        spaces += 1
    i += 1
print("Spaces:", spaces)
```

---

## ✅ Problem 93: Print prime numbers less than 100 using while

### 🧠 Description:

List all prime numbers below 100.

### 💡 Code:

```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True

n = 2
while n < 100:
    if is_prime(n):
        print(n, end=' ')
    n += 1
```

---

## ✅ Problem 94: Keep asking for a password until correct one is entered

### 🧠 Description:

Simple password check using `while`.

### 💡 Code:

```python
password = "python"
while True:
    entry = input("Enter password: ")
    if entry == password:
        print("Welcome!")
        break
```

---

## ✅ Problem 95: Simulate ATM PIN entry (3 attempts max)

### 🧠 Description:

Ask for PIN; lock after 3 wrong tries.

### 💡 Code:

```python
pin = "4321"
tries = 0
while tries < 3:
    entry = input("Enter PIN: ")
    if entry == pin:
        print("Access granted.")
        break
    else:
        print("Wrong PIN.")
        tries += 1
else:
    print("Card blocked.")
```

---

## ✅ Problem 96: Simulate dice rolls until a 6 is rolled

### 🧠 Description:

Keep rolling until 6 appears.

### 💡 Code:

```python
import random
roll = 0
while roll != 6:
    roll = random.randint(1, 6)
    print("Rolled:", roll)
```

---

## ✅ Problem 97: Generate a sequence like 1, 2, 4, 8, 16... up to n

### 🧠 Description:

Print exponential powers of 2 until it exceeds `n`.

### 💡 Code:

```python
n = 100
val = 1
while val <= n:
    print(val, end=' ')
    val *= 2
```

---

## ✅ Problem 98: Find the sum of positive numbers until a negative is entered

### 🧠 Description:

Add inputs; stop if user enters negative.

### 💡 Code:

```python
total = 0
while True:
    num = int(input("Enter number: "))
    if num < 0:
        break
    total += num
print("Sum:", total)
```

---

## ✅ Problem 99: Count how many times a digit appears in a number

### 🧠 Description:

Count occurrences of a specific digit.

### 💡 Code:

```python
n = 12233445
digit = 3
count = 0
while n > 0:
    if n % 10 == digit:
        count += 1
    n //= 10
print("Count of 3:", count)
```

---

## ✅ Problem 100: Keep dividing a number by 3 until it’s less than 1

### 🧠 Description:

Repeat division and track steps.

### 💡 Code:

```python
n = 100
while n >= 1:
    print(n)
    n /= 3
```

---




