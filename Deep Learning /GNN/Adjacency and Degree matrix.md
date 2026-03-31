

---

# 🔗 **Adjacency Matrix (A)**

## 🧠 কী?

Adjacency matrix হলো একটি **matrix (টেবিল)** যেখানে:

* কোন node (vertex) কোন node-এর সাথে connected (edge আছে কিনা) তা দেখানো হয়

---

## 📌 Formula (ছবিতে দেওয়া):

👉

* ( A_{ij} = 1 ) → যদি node i এবং j এর মধ্যে edge থাকে
* ( A_{ij} = 0 ) → যদি edge না থাকে

---

## 🔍 ছবির graph বুঝি

Graph এ node গুলো:
👉 1 — 2 — 3 — 4 (chain structure)

Connections:

* 1 ↔ 2
* 2 ↔ 3
* 3 ↔ 4

---

## 📊 Adjacency Matrix হবে:

|   | 1 | 2 | 3 | 4 |
| - | - | - | - | - |
| 1 | 0 | 1 | 0 | 0 |
| 2 | 1 | 0 | 1 | 0 |
| 3 | 0 | 1 | 0 | 1 |
| 4 | 0 | 0 | 1 | 0 |

---

## 🧠 সহজভাবে:

👉 “কে কার সাথে connected” — সেটার map

---

# 📊 **Degree Matrix (D)**

## 🧠 কী?

Degree matrix হলো একটি **diagonal matrix**
👉 যেখানে প্রতিটি node-এর **degree (কত edge আছে)** লেখা থাকে

---

## 🔍 ছবির graph অনুযায়ী degree:

* Node 1 → 1 edge
* Node 2 → 2 edges
* Node 3 → 2 edges
* Node 4 → 1 edge

---

## 📊 Degree Matrix হবে:

[
D =
\begin{bmatrix}
1 & 0 & 0 & 0 \
0 & 2 & 0 & 0 \
0 & 0 & 2 & 0 \
0 & 0 & 0 & 1
\end{bmatrix}
]

---

## 🧠 সহজভাবে:

👉 “প্রতিটি node কয়টা connection আছে” — সেটার matrix

---

# 🔥 **Main Difference (Important)**

| বিষয়      | Adjacency Matrix           | Degree Matrix     |
| --------- | -------------------------- | ----------------- |
| কী দেখায়  | connection (edge আছে কিনা) | connection সংখ্যা |
| value     | 0 বা 1                     | degree সংখ্যা     |
| structure | full matrix                | diagonal matrix   |

---

# 🎯 **GNN-এ কেন গুরুত্বপূর্ণ?**

👉 GNN-এ এই দুইটা matrix ব্যবহার হয়:

* graph structure বুঝতে
* node update করতে

👉 অনেক সময় ব্যবহার হয়:
[
A + I \quad \text{বা} \quad D^{-1}A
]

---

# 🧠 Viva Ready Line

👉
**“Adjacency matrix represents connectivity between nodes, while degree matrix stores the number of connections per node.”**

---

