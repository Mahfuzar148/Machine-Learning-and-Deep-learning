

Here is a **complete guide with official documentation and tutorials** for setting up and using **Jupyter Notebook**, **VS Code**, and **Google Colab** for Python programming—especially useful for ML, DL, and Data Science.

---

## 🧪 1. Jupyter Notebook Setup (Locally)

Jupyter is ideal for writing and executing Python code with visual outputs like charts or tables.

### 🔧 Installation (Recommended via Anaconda)

#### ➤ Option 1: Install Anaconda (recommended for beginners)

* Download from: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
* After installation, open Anaconda Navigator and launch **Jupyter Notebook**.

#### ➤ Option 2: Install using pip

```bash
pip install notebook
jupyter notebook
```

### 📘 Official Docs & Tutorials:

* [Jupyter Notebook Beginners Guide](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html)
* [JupyterLab vs. Notebook](https://jupyter.org/)
* [Interactive Jupyter Tutorial by DataQuest](https://www.dataquest.io/blog/jupyter-notebook-tutorial/)

### ✅ Sample Workflow

```python
# In a notebook cell
import numpy as np
print("Jupyter is running!")
```

You can also use markdown for notes, images, equations, etc.

---

## 🧑‍💻 2. VS Code for Python & Jupyter

### 🔧 Installation:

1. Download: [https://code.visualstudio.com/](https://code.visualstudio.com/)
2. Install the **Python Extension** from Microsoft (search “Python” in Extensions).
3. Optionally install the **Jupyter Extension** for running `.ipynb` files.

### 📘 Official Docs:

* [Python in VS Code](https://code.visualstudio.com/docs/python/python-tutorial)
* [Jupyter in VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)

### ✅ Sample Setup

* Create a `.py` file or `.ipynb` notebook.
* Use:

  ```python
  print("VS Code running Python")
  ```
* VS Code integrates with Git, virtual environments, and Conda too.

---

## ☁️ 3. Google Colab (Cloud-based Jupyter)

Colab is hosted by Google and requires no local installation. You can use GPU/TPU for free.

### 🔧 Access:

* Go to [https://colab.research.google.com](https://colab.research.google.com)
* Login with a Google account.
* Start a new notebook (File → New notebook)

### 📘 Official Docs & Help:

* [Colab User Guide (Google)](https://research.google.com/colaboratory/faq.html)
* [Colab GitHub Integration](https://colab.research.google.com/github/)

### ✅ Features:

* Supports Markdown + Code
* Upload datasets or connect to Google Drive
* Use GPU via:
  `Runtime → Change runtime type → Hardware Accelerator: GPU`

### ✅ Sample Colab Cell:

```python
import matplotlib.pyplot as plt
plt.plot([1,2,3,4],[10,20,25,30])
plt.title("Simple Plot in Colab")
plt.show()
```

---

## 🧰 Quick Comparison

| Feature       | Jupyter Notebook        | VS Code                    | Google Colab              |
| ------------- | ----------------------- | -------------------------- | ------------------------- |
| Installation  | Local or via Anaconda   | VS Code + Python extension | No install, browser-based |
| Suitable for  | ML, plotting, analysis  | Coding + debugging         | Quick ML, cloud computing |
| GPU Support   | Local GPU               | Local GPU                  | Free GPU/TPU from Google  |
| Collaboration | Moderate (export/share) | GitHub integration         | Live sharing, Drive       |
| Best For      | Hands-on learning       | Professional development   | Beginners, ML learners    |

---


