
---

## **Evaluation Loop — লাইনে লাইনে ব্যাখ্যা**

ধরা যাক আমাদের ভ্যালিডেশন বা টেস্ট লুপ এই রকম:

```python
def evaluate(model, device, data_loader, criterion):
    model.eval()  # 1
    total_loss = 0.0  # 2
    correct = 0        # 3
    total = 0          # 4

    with torch.no_grad():  # 5
        for data, targets in data_loader:  # 6
            data, targets = data.to(device), targets.to(device)  # 7
            outputs = model(data)  # 8
            loss = criterion(outputs, targets)  # 9
            total_loss += loss.item()  # 10

            preds = outputs.argmax(dim=1)  # 11
            correct += (preds == targets).sum().item()  # 12
            total += targets.size(0)  # 13

    avg_loss = total_loss / len(data_loader)  # 14
    accuracy = correct / total  # 15
    return avg_loss, accuracy  # 16
```

---

### **1. `model.eval()`**

* মডেলকে **evaluation mode**-এ সেট করে।
* Dropout, BatchNorm-এর মতো লেয়ারগুলো এখন deterministic আচরণ করবে।
* যদি এটা না দাও, তাহলে evaluation-এর সময় র‍্যান্ডম Dropout/BatchNorm আপডেটের কারণে ভুল রেজাল্ট আসবে।
* **সতর্কতা**: ট্রেনিং শেষে বা টেস্টের আগে অবশ্যই এই লাইনটা দিতে হবে।

---

### **2. `total_loss = 0.0`**

* সমস্ত ব্যাচের লস জমা রাখার জন্য কাউন্টার।
* শেষে গড় বের করা হবে (`avg_loss`)।
* যদি এটা ভুলে ইনিশিয়ালাইজ না করো, তাহলে আগের ভ্যালু জমে যাবে।

---

### **3. `correct = 0`**

* মোট সঠিক প্রেডিকশনের সংখ্যা রাখবে।

---

### **4. `total = 0`**

* মোট প্রেডিকশন সংখ্যার কাউন্টার।
* accuracy বের করার জন্য দরকার।

---

### **5. `with torch.no_grad():`**

* PyTorch-কে বলে দেয় **গ্রেডিয়েন্ট ক্যালকুলেশন বন্ধ রাখতে**।
* মেমরি ও কম্পিউট সময় বাঁচে, কারণ evaluation-এ ব্যাকপ্রপ দরকার হয় না।
* না দিলে অযথা গ্রেডিয়েন্ট জমা হবে।

---

### **6. `for data, targets in data_loader:`**

* DataLoader থেকে প্রতিটি ব্যাচ আনা।
* এখানে:

  * **`data`** → ইনপুট ইমেজ/ফিচারস
  * **`targets`** → সত্যিকারের লেবেল
* `batch_size` বড় হলে লুপের ইটারেশন কম হয়, কিন্তু মেমরি বেশি লাগে।

---

### **7. `data, targets = data.to(device), targets.to(device)`**

* মডেল আর ডেটা একই ডিভাইসে (CPU/GPU) আনতে হবে।
* যদি মডেল GPU-তে হয় কিন্তু ডেটা CPU-তে থাকে, তাহলে `RuntimeError` হবে।

---

### **8. `outputs = model(data)`**

* ফরওয়ার্ড পাস করে প্রেডিকশন বের করা।
* Classification হলে shape হবে `[batch_size, num_classes]`।

---

### **9. `loss = criterion(outputs, targets)`**

* লস ফাংশন (যেমন CrossEntropyLoss) দিয়ে লস বের করা।
* ভ্যালিডেশন লস মডেলের জেনারালাইজেশন ট্র্যাক করতে সাহায্য করে।

---

### **10. `total_loss += loss.item()`**

* `.item()` → টেনসর থেকে scalar float বের করে নেয়।
* সব ব্যাচের লস যোগ হয়, পরে গড় বের করব।

---

### **11. `preds = outputs.argmax(dim=1)`**

* **`argmax`** → সবচেয়ে বড় ভ্যালুর ইনডেক্স নেয়, যেটি predicted class।
* `dim=1` কারণ প্রতিটি row (sample)-তে classes থাকে।

---

### **12. `correct += (preds == targets).sum().item()`**

* প্রেডিকশন আর আসল লেবেল মিলে গেলে `True` (1), না মিললে `False` (0)।
* `.sum()` দিয়ে ব্যাচে কয়টা সঠিক হয়েছে তা বের হয়, `.item()` দিয়ে float/ইন্টে কনভার্ট।

---

### **13. `total += targets.size(0)`**

* ব্যাচে কয়টা স্যাম্পল আছে সেটা যোগ করা।
* Accuracy বের করতে দরকার।

---

### **14. `avg_loss = total_loss / len(data_loader)`**

* মোট লসকে ব্যাচের সংখ্যা দিয়ে ভাগ করে গড় লস পাওয়া যায়।

---

### **15. `accuracy = correct / total`**

* মোট সঠিক প্রেডিকশনের অনুপাত।

---

### **16. `return avg_loss, accuracy`**

* গড় লস আর অ্যাকুরেসি একসাথে ফেরত দেয়।
* এগুলো লুপ শেষে লগ বা প্রিন্ট করে মডেলের পারফরম্যান্স দেখা হয়।

---

## **📌 প্যারামিটার পরিবর্তনে প্রভাব**

* **`batch_size`** → বড় হলে দ্রুত কিন্তু মেমরি বেশি; ছোট হলে স্লো কিন্তু কম মেমরি।
* **`criterion`** পরিবর্তন করলে লস ক্যালকুলেশন পদ্ধতি বদলায় (যেমন MSE vs CrossEntropy)।
* **`argmax(dim=1)`** → dim ভুল দিলে ভুল অ্যাকুরেসি আসবে।
* **`no_grad()`** বাদ দিলে মেমরি লিক/স্লো ট্রেনিং।

---

