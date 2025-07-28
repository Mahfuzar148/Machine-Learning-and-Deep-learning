
---

# ⚙️ **Installing PyTorch (CPU/GPU, pip/conda)**

---

## 🔧 Step 1: Check Your System

Before installation, make sure:

* Python 3.8 – 3.12 is installed (`python --version`)
* pip or conda is available (`pip --version` or `conda --version`)
* Optional for GPU: You have an NVIDIA GPU and the proper CUDA drivers installed.

---

## 📦 Option 1: Using `pip`

### ▶️ CPU-Only Installation

```bash
pip install torch torchvision torchaudio
```

### ▶️ GPU Installation (CUDA 11.8)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### ▶️ GPU Installation (CUDA 12.1)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> ℹ️ You don’t need to manually install CUDA. PyTorch includes the binaries.

---

## 📦 Option 2: Using `conda`

### ▶️ CPU-Only Installation

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### ▶️ GPU Installation (CUDA 12.1)

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

> ✅ `pytorch-cuda=12.1` automatically installs the required CUDA Toolkit if compatible with your drivers.

---

## 💡 Optional: Create a Conda Virtual Environment First

```bash
conda create --name pytorch-env python=3.10
conda activate pytorch-env
```

Then proceed with the above conda installation steps.

---

## ✅ Step 3: Verify Installation

```python
import torch
print(torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
```

---

## 🧰 Extra Notes

| Method  | Pros                                    | Cons                                 |
| ------- | --------------------------------------- | ------------------------------------ |
| `pip`   | Easy, flexible                          | Manual CUDA management may be tricky |
| `conda` | Handles dependencies/CUDA automatically | Slightly larger install size         |

---

## 🔗 Official PyTorch Install Tool

For a tailored command based on your system:
👉 [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)

---


