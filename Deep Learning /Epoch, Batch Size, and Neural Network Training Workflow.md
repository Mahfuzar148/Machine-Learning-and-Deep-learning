
---

# 🧠 Epoch, Batch Size, and Neural Network Training Workflow

---

## 🔁 What is an **Epoch**?

An **epoch** is **one complete pass** through the **entire training dataset** by the neural network.

During training:

* The model looks at every sample once per epoch.
* With each epoch, the model **adjusts weights** based on the **loss gradients** to improve predictions.

---

## ⚙️ Key Steps in Each Epoch

| Step                          | Description                                                                                                          |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **1. Forward Pass**           | Inputs are passed through the model to generate predictions.                                                         |
| **2. Loss Calculation**       | Compares predicted values to actual targets using a **loss function**.                                               |
| **3. Backpropagation**        | Uses the **gradient of the loss** to calculate how each weight affects the error (via **chain rule** from calculus). |
| **4. Weight Update**          | Uses optimization algorithms like **Gradient Descent, Adam, RMSprop** to update weights to reduce the loss.          |
| **5. Repeat for All Batches** | The above steps repeat for each batch until the entire dataset is processed.                                         |

---

## 📦 What is **Batch Size**?

* **Batch size** determines how many **samples** the model sees before updating weights.
* Instead of using the **whole dataset** at once (which can be slow or memory-intensive), it is split into **smaller batches**.

---

### 🔍 Why Use Batches?

* It improves **training efficiency** and **generalization**.
* Smaller batches → more updates per epoch.
* Larger batches → fewer updates but might be more stable.

---

## 🔁 Epochs vs. Batches — Example

Suppose:

* Dataset = 1000 samples
* **Batch size** = 100
* Then in **1 epoch**, there are 1000 / 100 = **10 weight updates**
* If **epochs = 10**, then the model sees the entire dataset **10 times** (i.e., **10 × 10 = 100 updates total**)

---

## 🧠 Final Takeaway:

| Term           | Meaning                                                       |
| -------------- | ------------------------------------------------------------- |
| **Epoch**      | One complete pass over the training data                      |
| **Batch Size** | Number of training examples used in one forward/backward pass |
| **Iteration**  | One weight update (i.e., 1 batch)                             |


---

Would you like help visualizing this with a diagram or Python training loop example?
