![Image](https://images.openai.com/static-rsc-4/SSi7U98nUEkS0B6hAuUuGyEVKmW2_0AQeA3Afk9vTXgjx4RJtQWOTZOJLW4dUKvhteZwtfYy7t4KS_IMCItCxdGv0NI39pVYp9Acrrqj27NTJh7UsdkRlfFC7ZxxS4CZNUd8_4KcNMy2slxVYTfaFNT9c4-umW6E9P9o2jHWar1RGa3owig-lFIsySrT3rSm?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/_SPjc4V4jMG1fAZB6k_wWbLMjNzbZLsp_CXvrBpt-nlIab2ObBtZLAPObod1JqU8y7xVLeUEedAzdNj0Wn6cV5Y9rAXRLOlouIQPa1BiO6hxyGzc3gedA02L8ceQqFSFy36lABQ-nJkrkEshSlJEb5Wi7rUqCO1ReDDmpZogLqqhZdUZr_m0wPqSG0ulNdzw?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/WBqWqY8tlSoNyYZsDNMX0R_qQQDjHvHEnIiRwncrYIulb9J4rsRqjga3CJ70YyjHWD5I_JrXo_075WhpCZSvz6Zby_rXgx_BMvwTesG0p838LJfhSv7_EPm_0tSm92iBPtuIz6pDCBIZeT6NkUcB-mbz26dL0MPeIW_y5BKk5KgNoTLn0XiXOqZU4LPFzkk-?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/WCyOElCg4HiUTfvB2W4pck8ALPvYJ0NubK2OKjioa3UGuyFa-T2snXq5IYwuySFL-ICazbgR5Huwhr6KA15MvP6iMUOWcvlyfHgnRDg6LcprP0fjGt4wWdokK5h6eVn0J1Hlz4mo7u-Z6dP6sxGqB6SOheNaB08lQGchevPkc1j_Wtzf7Re7i92DbcpQvM6F?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/WEs_uVY1xSfoK0SXnbvzIMcRWrPl70u0WZYlSeMRnGdkBI1zH3It64yKoA_2YxRxxu_B5FrTVKW7A4IUXi7XKyUetUyKmWe6REsH0PYL-FimQ2q2BVhv9aWZfBwRiQ9TaTUCfmREVgDEUxoTB5ntwZaSldKAY82Q6JvXjJ0DEF7PpPShmF2oZwNHE6pXgtHn?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/QoAcZzT0AL6SaushW4gxeLmCY1plgZ74mflGNlHuC6pG9T5cXJu5i7IKkITdfPPgL47lNi1cwQnuF7MoCxjeE1NLr8Eg9mAiDU4Q304UpJXJXMaRsjUZUFQ3xtTmgs5CjRQMr6YlhWJDIlgxMeEX_cwedSvzth1n7FKm3wiqIhz3rnY4BXYtUb87shh_7mZS?purpose=fullsize)

## 📊 লিনিয়ার রিগ্রেশন (Linear Regression) — সহজ ভাষায় ব্যাখ্যা

লিনিয়ার রিগ্রেশন হলো একটি **স্ট্যাটিস্টিক্যাল এবং মেশিন লার্নিং পদ্ধতি**, যার মাধ্যমে আমরা একটি সরল রেখা (straight line) ব্যবহার করে ডেটার মধ্যে সম্পর্ক বুঝতে পারি এবং ভবিষ্যৎ মান (prediction) অনুমান করতে পারি।

---

## 🧮 মূল সমীকরণ:

**y = mx + c**

এখানে প্রতিটি অংশের অর্থ:

### 🔹 y (Dependent Variable)

* যেটা আমরা **predict বা অনুমান করতে চাই**
* ছবিতে: *Ending Price*

### 🔹 x (Independent Variable)

* যেটার উপর ভিত্তি করে আমরা y বের করি
* ছবিতে: *Starting Price*

---

## 📈 m (Slope) — রেখার ঢাল

* m দেখায় x পরিবর্তন হলে y কতটা পরিবর্তিত হয়
* সূত্র:
  **m = Δy / Δx**
* যদি m বেশি হয় → রেখা বেশি খাড়া
* যদি m কম হয় → রেখা তুলনামূলক সমতল

👉 সহজভাবে:
x এক ইউনিট বাড়লে y কত বাড়ে — সেটাই slope

---

## 📍 c (Intercept)

* যখন x = 0, তখন y এর মান
* এটাকে বলে **y-intercept**
* গ্রাফে এটি সেই পয়েন্ট যেখানে রেখাটি y-অক্ষকে কাটে

---

## 📊 গ্রাফের ব্যাখ্যা

### 🔸 ডট (Orange points)

* আসল ডেটা (Actual values)

### 🔸 নীল ড্যাশড লাইন

* রিগ্রেশন লাইন (Prediction line)

### 🔸 কালো ড্যাশড লাইন (Residuals / Error)

* আসল মান (actual y) এবং অনুমিত মান (predicted y) এর পার্থক্য

👉 অর্থাৎ:
**Error = Actual value - Predicted value**

---

## 🎯 লিনিয়ার রিগ্রেশনের লক্ষ্য

একটি এমন সরল রেখা বের করা যাতে:

* সব ডেটা পয়েন্টের কাছাকাছি থাকে
* Error (ভুল) যত কম হয়

---

## 📌 বাস্তব জীবনের উদাহরণ

### 🏠 বাড়ির দাম অনুমান

* x = বাড়ির আকার (sq ft)
* y = বাড়ির দাম

👉 বড় বাড়ি → বেশি দাম
👉 এই সম্পর্ককে লিনিয়ার রিগ্রেশন দিয়ে বোঝা যায়

---

## 🧠 সংক্ষেপে

* লিনিয়ার রিগ্রেশন = সরল রেখা দিয়ে সম্পর্ক বোঝা
* সমীকরণ: **y = mx + c**
* m = পরিবর্তনের হার (slope)
* c = শুরুর মান (intercept)
* লক্ষ্য = prediction করা

---

