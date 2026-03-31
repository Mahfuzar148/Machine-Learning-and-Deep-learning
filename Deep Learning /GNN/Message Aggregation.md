
---

# 📥 1️⃣ **Message Aggregation (Neighbour থেকে তথ্য নেওয়া)**

## 🧠 ধারণা:

প্রতিটি node তার পাশের node (neighbours) থেকে তথ্য সংগ্রহ করে

## 🔢 Formula:

[
h_v^{agg} = \sum_{u \in N(v)} h_u
]

👉 এখানে:

* (v) = target node
* (N(v)) = neighbour set
* (h_u) = neighbour node-এর feature

---

## 📌 কী হয়?

👉 node v তার সব neighbour-এর feature যোগ করে একটি নতুন vector বানায়

---

## 🔍 উদাহরণ:

ধরা যাক:

* Node 2 → feature = 3
* Node 4 → feature = 5

👉 তাহলে Node 3:
[
h_3^{agg} = 3 + 5 = 8
]

---

## 🧠 সহজভাবে:

👉 “পাশের node থেকে তথ্য নিয়ে একসাথে করা”

---

# 🔄 2️⃣ **Transformation (Learnable weight দিয়ে পরিবর্তন)**

## 🧠 ধারণা:

Aggregation করার পর feature vector-কে model transform করে

## 🔢 Formula:

[
h_v^{trans} = W \cdot h_v^{agg}
]

👉 এখানে:

* (W) = weight matrix (learnable parameter)

---

## 📌 কী হয়?

👉 model শিখে কোন feature গুরুত্বপূর্ণ
👉 সেই অনুযায়ী feature modify করে

---

## 🔍 উদাহরণ:

ধরা যাক:

* (h_v^{agg} = 8)
* (W = 0.5)

👉 তাহলে:
[
h_v^{trans} = 0.5 \times 8 = 4
]

---

## 🧠 সহজভাবে:

👉 “তথ্য নিয়ে সেটাকে smarter বানানো”

---

# ⚖️ 3️⃣ **Normalization (Balance রাখা)**

## 🧠 সমস্যা:

👉 যদি কোনো node-এর অনেক neighbour থাকে
👉 তাহলে তার influence বেশি হয়ে যাবে

---

## 🔢 Formula:

[
\hat{A} = D^{-1/2}(A + I)D^{-1/2}
]

👉 এখানে:

* (A) = adjacency matrix
* (I) = self-loop
* (D) = degree matrix

---

## 📌 কী হয়?

👉 সব node-এর influence balance করা হয়

---

## 🎯 কেন দরকার?

* high-degree node dominate করতে পারবে না
* training stable হয়

---

## 🧠 সহজভাবে:

👉 “সব node-কে equal importance দেওয়া”

---

# 🔥 **সব একসাথে (Full GCN Equation)**

[
H' = \sigma(\hat{A} H W)
]

---

## 🧠 Step Flow:

1. **Normalization** → graph balance করা
2. **Aggregation** → neighbour info নেওয়া
3. **Transformation** → weight apply
4. **Activation** → output তৈরি

---

# 🔑 **Final Summary**

| Step           | কাজ                     |
| -------------- | ----------------------- |
| Aggregation    | neighbour থেকে info নেয় |
| Transformation | weight দিয়ে modify করে  |
| Normalization  | influence balance করে   |

---

# 🧠 Viva Ready Line

👉
**“GCN updates node features by aggregating neighbor information, transforming it using learnable weights, and normalizing to ensure balanced contribution.”**

---

