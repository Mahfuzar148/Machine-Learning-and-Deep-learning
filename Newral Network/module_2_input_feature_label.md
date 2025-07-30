

---




# 🎓 Module 2: ইনপুট, ফিচার, লেবেল (Input, Feature, Label)

---

## 🧩 1. ইনপুট (Input) কী?

**ইনপুট** হলো সেই ডেটা, যা আমরা মডেলে পাঠাই যেন সেটা শেখে এবং সঠিক সিদ্ধান্ত নিতে পারে।

👉 ইনপুট হতে পারে:
- একটি সংখ্যা (যেমন: ছাত্রের রেজাল্ট)
- একটি ছবি (যেমন: বিড়াল না কুকুর)
- একটি শব্দ (যেমন: ভয়েস কমান্ড)
- একটি লেখা (যেমন: movie review)

🎯 **মেশিন ইনপুট হিসেবে কেবল সংখ্যা বোঝে। তাই ছবি, লেখা, শব্দ — সবকিছু সংখ্যায় রূপান্তর করে দিতে হয়।**

### 🖼️ উদাহরণ: হাতের লেখা ডিজিট (MNIST)
প্রতিটি 28x28 পিক্সেল এর ছবি একেকটি ইনপুট।

![MNIST Input](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

🔗 Image Source: [Wikipedia - MNIST](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

---

## 🧬 2. ফিচার (Feature) কী?

**ফিচার** হলো ইনপুটের সেই অংশ বা গুণাবলি যেগুলো বিশ্লেষণ করে মডেল সিদ্ধান্ত নেয়।

> তুমি যেভাবে কারো উচ্চতা, কথা বলার ভঙ্গি, পোশাক দেখে তাকে চেনো — ঠিক তেমনিভাবে মেশিনও ফিচার ব্যবহার করে।

📌 উদাহরণ:
- ছবির ক্ষেত্রে → প্রতিটি পিক্সেল = একটি ফিচার
- ছাত্রের ক্ষেত্রে → বয়স, রেজাল্ট, উচ্চতা = ফিচার
- টেক্সটের ক্ষেত্রে → প্রতিটি শব্দ বা শব্দের সংখ্যা = ফিচার

### 🎯 উদাহরণ: ডিজিট ফিচার ক্লাস্টার
নিচের চিত্রে বিভিন্ন ডিজিট (0–9) কীভাবে ভিন্ন ভিন্ন ফিচার স্পেসে অবস্থান করছে তা দেখানো হয়েছে।

![Digit Feature & Cluster](https://raw.githubusercontent.com/amitrajitbose/Visualizing-MNIST-using-t-SNE/master/mnist_plot.png)

🔗 Source: [MNIST Cluster Visualization](https://raw.githubusercontent.com/amitrajitbose/Visualizing-MNIST-using-t-SNE/master/mnist_plot.png)

---

## 🏷️ 3. লেবেল (Label) কী?

**লেবেল** হলো সেই **সঠিক উত্তর**, যেটা মডেল শেখার সময় ইনপুটের সাথে দেওয়া হয়।

যদি ইনপুট হয় একটি কুকুরের ছবি, তাহলে তার লেবেল হবে `"Dog"`।

> মডেল শেখে ইনপুট থেকে কীভাবে লেবেল তৈরি করতে হয়।

📌 লেবেল হতে পারে:
- একটি সংখ্যা (যেমন: ৭, ৩.৫)
- একটি ক্লাস (যেমন: "Cat", "Dog", "Spam")

### 🖼️ উদাহরণ: বিড়াল না কুকুর ছবি

![Cat vs Dog Input and Label](https://miro.medium.com/v2/resize:fit:1200/1*I72FwzMNtALkUDrU3DRQbQ.png)

🔗 Source: [Medium - Dog vs Cat Classification](https://miro.medium.com/v2/resize:fit:1200/1*I72FwzMNtALkUDrU3DRQbQ.png)

---

## 🔄 4. ইনপুট → ফিচার → লেবেল → শেখা

মডেল শেখে কীভাবে ইনপুট থেকে ফিচার বের করে সঠিক লেবেল তৈরি করতে হয়।  
এটা অনেকটা এইরকম:

```

\[ছবি] → ফিচার বিশ্লেষণ → শেখা → "এটা বিড়াল"

```

🎯 প্রতিবার ভুল করলে, মডেল নিজের ভিতরের ওজন (weights) আপডেট করে যেন পরেরবার সঠিক উত্তর দিতে পারে।

---

## 🧠 5. একটি বাস্তব উদাহরণ

**উদাহরণ: একটি ডিজিট (৯) চেনা**

- ইনপুট = 28x28 পিক্সেল ডিজিট ছবি
- ফিচার = 784টি পিক্সেল মান
- লেবেল = `9`

👉 মডেল শেখে: “এই ধরণের ফিচার মান মানে = 9”

---

## 📊 6. ইনপুট থেকে আউটপুট — পূর্ণ ফ্লো চিত্র

![Input to Output Flow](https://raw.githubusercontent.com/ageron/handson-ml/master/images/ai_overview_diagram.png)

🔗 Source: [Hands-on ML Book Diagram](https://raw.githubusercontent.com/ageron/handson-ml/master/images/ai_overview_diagram.png)

---

## ✅ সংক্ষেপে টেবিল

| টার্ম | মানে | উদাহরণ |
|------|------|---------|
| Input | যা মডেলে যায় | ছবি, সংখ্যা, লেখা |
| Feature | ইনপুটের গঠন | বয়স, উচ্চতা, পিক্সেল |
| Label | সঠিক উত্তর | Cat, 7, Positive |

---

## 📚 Module 2 শেষ!

তুমি এখন জানো:
- ইনপুট কী
- ফিচার কী
- লেবেল কী
- এবং মডেল কীভাবে এগুলো থেকে শেখে

---



---


