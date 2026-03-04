**Dataset inspection** মানে হলো কোনো **dataset (ডেটাসেট)** ব্যবহার করার আগে সেটাকে ভালোভাবে **দেখা, বোঝা ও পরীক্ষা করা**। অর্থাৎ ডেটার ভিতরে কী আছে, কেমন ধরনের ডেটা আছে, কোনো ভুল বা missing value আছে কিনা—এসব যাচাই করাকে dataset inspection বলে।

সহজভাবে বললে 👉 **ডেটা ব্যবহার করার আগে ডেটাকে পর্যবেক্ষণ করা।**

### Dataset inspection-এ সাধারণত যে কাজগুলো করা হয়

1. **ডেটার আকার দেখা**

   * কতগুলো row (record) আছে
   * কতগুলো column (feature) আছে

2. **ডেটার টাইপ দেখা**

   * সংখ্যা (numeric)
   * লেখা (string/text)
   * category ইত্যাদি

3. **Missing value আছে কিনা দেখা**

   * কোনো ঘরে data ফাঁকা আছে কি না

4. **ডেটার sample দেখা**

   * প্রথম কয়েকটা row দেখে বোঝা ডেটা কেমন

5. **ভুল বা অস্বাভাবিক data আছে কিনা চেক করা**

   * যেমন: বয়স 200 লেখা থাকলে সেটা ভুল

### উদাহরণ (Python pandas এ)

```python
import pandas as pd

df = pd.read_csv("data.csv")

print(df.head())      # প্রথম ৫টা row দেখা
print(df.shape)       # row ও column সংখ্যা
print(df.info())      # data type দেখা
print(df.describe())  # statistical summary
```

📌 **সংক্ষেপে:**
Dataset inspection = **ডেটা বিশ্লেষণের আগে ডেটাকে বুঝে নেওয়ার প্রক্রিয়া।**

---

