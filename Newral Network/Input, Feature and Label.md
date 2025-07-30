
---

# 🎓 Module 2: ইনপুট, ফিচার, লেবেল

---

## 🧩 1. ইনপুট (Input) কী?

### ➤ ব্যাখ্যা:

ইনপুট হচ্ছে সেই **তথ্য**, যা আমরা মডেলকে দিই শেখার জন্য।
মডেল ইনপুট হিসেবে শুধু **সংখ্যা বোঝে**। ছবি, লেখা বা শব্দ — সবকিছুই সংখ্যায় রূপান্তর করে দিতে হয়।

### ➤ ইনপুট হতে পারে:

* ছবি (image)
* লেখা (text)
* সংখ্যা (numerical data)
* শব্দ (audio waveform)

### 🖼️ উদাহরণ: হাতের লেখা ডিজিট (MNIST)

![MNIST Input](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)
🔗 [MNIST Example Full Image](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

প্রতিটি **২৮ x ২৮ পিক্সেল** এর একটি **ছবি** ইনপুট — যেমন `0`, `1`, `2`, ... `9`

---

## 🧬 2. ফিচার (Feature) কী?

### ➤ ব্যাখ্যা:

**ফিচার** হলো ইনপুটের সেই অংশ যেখান থেকে মডেল শেখে।
ছবির প্রতিটি পিক্সেল, ছাত্রের বয়স-উচ্চতা-রেজাল্ট — এগুলো সব ফিচার।

> ➤ মডেল ফিচার থেকে বুঝে ফেলে: “এই ইনপুটের সঙ্গে কোন আউটপুট মিলবে।”

### 🎯 উদাহরণ:

ধরো ছাত্রের ডেটা:

```text
[ বয়স = 16, উচ্চতা = 5.6 ft, রেজাল্ট = 87 ]
```

এখানে তিনটি ফিচার আছে: `[16, 5.6, 87]`

### 🖼️ ফিচার বুঝতে একটি ক্লাস্টার চিত্র:

![Feature & Label Visual](https://raw.githubusercontent.com/amitrajitbose/Visualizing-MNIST-using-t-SNE/master/mnist_plot.png)
🔗 [Digit Feature Visualization](https://raw.githubusercontent.com/amitrajitbose/Visualizing-MNIST-using-t-SNE/master/mnist_plot.png)

এখানে তুমি দেখতে পাচ্ছো বিভিন্ন ডিজিট `0-9` কীভাবে একে অপর থেকে আলাদা ফিচারে বিভক্ত হয়েছে।

---

## 🏷️ 3. লেবেল (Label) কী?

### ➤ ব্যাখ্যা:

**লেবেল** হলো সঠিক উত্তর — যেটা মডেল শেখার সময় জানে।
মডেল শিখে ইনপুট দেখে কীভাবে লেবেল বানানো যায়।

### 🎯 উদাহরণ:

* ইনপুট: 28x28 একটি ডিজিট ছবি

* লেবেল: 7

* ইনপুট: বিড়ালের ছবি

* লেবেল: `Cat`

* ইনপুট: গাড়ির ছবি

* লেবেল: `Car`

### 🖼️ উদাহরণ — বিড়াল না কুকুর?

![Cat vs Dog Input and Label](https://miro.medium.com/v2/resize\:fit:1200/1*I72FwzMNtALkUDrU3DRQbQ.png)
🔗 [Cat vs Dog Image](https://miro.medium.com/v2/resize:fit:1200/1*I72FwzMNtALkUDrU3DRQbQ.png)

এখানে ইনপুট হলো `ছবি`, আর লেবেল হলো → `Dog` বা `Cat`

---

## 🔄 4. ইনপুট → ফিচার → লেবেল → শেখা

### ➤ ব্যাখ্যা:

মডেল শেখে কিভাবে ইনপুট থেকে ফিচারগুলো বের করে লেবেল বানাতে হয়।

> এটা অনেকটা ছাত্রকে প্রশ্ন দিয়ে উত্তর শেখানোর মতো।

---

## 🧠 উদাহরণ দিয়ে পুরো প্রসেস:

### 📷 ধরো তুমি এই ছবি দাও:

![Digit 9](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f0/MnistExamples.png/220px-MnistExamples.png)

➡️ এটা মডেলের ইনপুট
➡️ ফিচার = ছবির প্রতিটি পিক্সেল (784টি সংখ্যা)
➡️ লেবেল = `9`

---

## 🧭 চিত্র: ইনপুট → মডেল → লেবেল

![Input to Output Flow](https://raw.githubusercontent.com/ageron/handson-ml/master/images/ai_overview_diagram.png)
🔗 [View Diagram](https://raw.githubusercontent.com/ageron/handson-ml/master/images/ai_overview_diagram.png)

---

## ✅ সংক্ষেপে:

| টার্ম   | ব্যাখ্যা                       |
| ------- | ------------------------------ |
| Input   | মডেলে ঢোকে — ছবি, সংখ্যা, লেখা |
| Feature | ইনপুটের বিশ্লেষণযোগ্য অংশ      |
| Label   | সঠিক উত্তর (মডেল যা শিখবে)     |

---

## 🎯 Module 2 শেষ!

এখন তুমি পুরোপুরি বুঝতে পেরেছো:

* ইনপুট কি?
* ফিচার কাকে বলে?
* লেবেল কিভাবে কাজ করে?

---

