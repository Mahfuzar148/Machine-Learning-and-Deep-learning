
---

# 📄 **Documentation: Conda Virtual Environment + PyTorch Installation**

## 🧰 Prerequisites

Make sure you have **Anaconda** or **Miniconda** installed. To check:

```bash
conda --version
```

If not installed, download from: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

---

## 📦 Step 1: Create a Conda Virtual Environment

```bash
conda create --name pytorch-env python=3.10
```

Replace `pytorch-env` with your preferred environment name, and `3.10` with your desired Python version.

---

## 🚀 Step 2: Activate the Environment

```bash
conda activate pytorch-env
```

Now you are inside the isolated environment.

---

## 🔧 Step 3: Install PyTorch

### ➤ For CPU-only version:

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### ➤ For GPU version with CUDA 12.1:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

> 📝 Replace `12.1` with your desired CUDA version. You must have compatible NVIDIA drivers installed.

---

## ✅ Step 4: Verify the Installation

Run Python and test:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # True = using GPU, False = CPU
```

---

## 🧹 Optional: Deactivate the Environment

```bash
conda deactivate
```

---

## 🗑 Optional: Remove the Environment

```bash
conda remove --name pytorch-env --all
```

---

## 📚 Additional Resources

* Official PyTorch Install Guide: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
* Conda Documentation: [https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

---

