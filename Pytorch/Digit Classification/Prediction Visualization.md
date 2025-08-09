
---

## **7. Prediction Visualization — লাইনে–লাইনে ব্যাখ্যা**

```python
def visualize_predictions(model, device, test_loader, n=6):
```

* **`model`**: যে ট্রেইন করা মডেল দিয়ে প্রেডিকশন দেখাবো।
* **`device`**: CPU/GPU ডিভাইস — মডেল ও ডেটা একই ডিভাইসে থাকা দরকার।
* **`test_loader`**: টেস্ট ডেটা ব্যাচে ইটারেট করার জন্য DataLoader।
* **`n`**: কয়টি ইমেজ দেখানো হবে (ডিফল্ট 6)। বাড়ালে গ্রাফ বড় হবে, কিন্তু স্ক্রিন স্পেস বেশি লাগবে।

---

```python
    model.eval()
```

* মডেলকে **evaluation mode**-এ দেয়।
* Dropout বন্ধ হয়, BatchNorm ফিক্সড mean/variance ব্যবহার করে।
* ভিজুয়ালাইজেশনের সময় সর্বদা `eval()` দরকার, না হলে প্রেডিকশন এলোমেলো হতে পারে।

---

```python
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
```

* `iter(test_loader)` → DataLoader থেকে একবারে ইটারেটর তৈরি।
* `next(data_iter)` → প্রথম ব্যাচ নেয়।
* নোট: বড় `batch_size` দিলে এখানে বেশি ছবি পাওয়া যাবে।

---

```python
    images, labels = images.to(device), labels.to(device)
```

* ইমেজ ও লেবেলকে একই ডিভাইসে পাঠায়, যাতে মডেল সেগুলো ব্যবহার করতে পারে।
* CPU↔GPU মিসম্যাচ হলে Error হবে।

---

```python
    outputs = model(images)
```

* মডেলের **forward pass** → raw logits/স্কোর বের করে।
* এই আউটপুট CrossEntropyLoss-compatible শেপে থাকে (`[batch, classes]`)।

---

```python
    _, preds = torch.max(outputs, 1)
```

* প্রতি স্যাম্পলের সর্বোচ্চ স্কোরের ক্লাস বের করে।
* দ্বিতীয় আর্গুমেন্ট `1` মানে row-wise (প্রতি স্যাম্পল) max নেবে।
* এখানে `_` ভেরিয়েবলটা শুধু max value রাখে (আমরা লাগাচ্ছি না), `preds` রাখে ক্লাস ইনডেক্স।

---

```python
    images = images.cpu()
    labels = labels.cpu()
    preds = preds.cpu()
```

* ভিজুয়ালাইজেশনের জন্য সবকিছুকে CPU-তে আনা হয়, কারণ `matplotlib` GPU টেনসর বুঝতে পারে না।
* CUDA টেনসর দিলে error দেবে।

---

```python
    plt.figure(figsize=(12, 4))
```

* একটি নতুন figure তৈরি, সাইজ `(width=12 inch, height=4 inch)`।
* `n` বাড়ালে height/width বাড়াতে হবে, নাহলে ছবিগুলো চাপা পড়বে।

---

```python
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"True: {labels[i]}\nPred: {preds[i]}")
        plt.axis('off')
```

* **`plt.subplot(1, n, i+1)`** → 1 row, `n` columns-এর সাবপ্লট তৈরি, যেখানে `i+1` তম প্লট সিলেক্ট।
* **`.squeeze()`** → এক্সট্রা ডাইমেনশন (যেমন \[1, 28, 28]) সরায়, যাতে matplotlib প্লট করতে পারে।
* **`cmap='gray'`** → গ্রেস্কেল কালারম্যাপ (MNIST-এর জন্য)। RGB হলে বাদ দিতে হবে।
* `plt.title` → শিরোনামে আসল লেবেল ও প্রেডিকশন দেখায়।
* `plt.axis('off')` → অক্ষরেখা সরিয়ে ফেলে, যাতে ছবি ক্লিন থাকে।

---

```python
    plt.tight_layout()
    plt.show()
```

* **`tight_layout()`** → সাবপ্লটগুলোর মধ্যে ফাঁকা ঠিক করে।
* **`show()`** → গ্রাফ দেখায়।

---

## **প্যারামিটার টিউনিং এর প্রভাব**

| প্যারামিটার                 | পরিবর্তন করলে কী হবে                                                        |
| --------------------------- | --------------------------------------------------------------------------- |
| `n`                         | ছবি সংখ্যা বাড়বে/কমবে। বড় `n`-এ figure সাইজও বাড়াও।                         |
| `figsize`                   | গ্রাফের ভিজুয়াল সাইজ পরিবর্তন হবে।                                          |
| `cmap`                      | কালারম্যাপ পাল্টে ভিজুয়াল ইফেক্ট বদলানো যাবে (যেমন `'viridis'`, `'plasma'`) |
| `batch_size` (test\_loader) | একবারে কত ছবি লোড হবে, তার উপর নির্ভর করে `n`-এর সর্বোচ্চ মান।              |

---

## **8. Run the Training, Test, and Visualization**

```python
train(model, device, train_loader, optimizer, criterion, epochs=5)
```

* `epochs=5` → ৫ বার পুরো ডেটাসেট দিয়ে মডেল ট্রেন হবে।
* বাড়ালে মডেল বেশি শিখবে কিন্তু overfitting ঝুঁকি বাড়তে পারে।
* কমালে underfitting হতে পারে।

---

```python
test(model, device, test_loader)
```

* ট্রেনিং শেষে টেস্ট সেটে মডেলের অ্যাকুরেসি/লস মাপা হবে।
* এখানে shuffle=False থাকা উচিত যাতে রেজাল্ট reproducible হয়।

---

```python
visualize_predictions(model, device, test_loader)
```

* কিছু টেস্ট স্যাম্পল নিয়ে আসল লেবেল বনাম মডেলের প্রেডিকশন দেখাবে।
* মডেলের ভুল/সঠিক কোথায় হচ্ছে সেটা বুঝতে সাহায্য করে।

---


