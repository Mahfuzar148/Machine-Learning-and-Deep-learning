Here are **all the main ways you can build a Fully Connected Neural Network (FCNN) with 5 hidden layers in PyTorch**, from simple to advanced. I’ll show multiple patterns so you understand the flexibility.

---

# 🔧 1. Basic FCNN using `nn.Sequential`

This is the simplest and most concise way.

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),

    nn.Linear(64, 128),
    nn.ReLU(),

    nn.Linear(128, 128),
    nn.ReLU(),

    nn.Linear(128, 64),
    nn.ReLU(),

    nn.Linear(64, 32),
    nn.ReLU(),

    nn.Linear(32, 1)  # output layer
)

x = torch.randn(5, 10)
output = model(x)
print(output.shape)
```

✔️ Good for quick prototyping
❌ Harder to customize later

---

# 🧱 2. Using a Custom `nn.Module` (Most Common)

Gives full control and readability.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.out(x)
        return x

model = FCNN()
```

✔️ Best balance of flexibility and clarity
✔️ Easy to debug and modify

---

# 🔁 3. Dynamic Layer Construction (Loop-based FCNN)

Useful when layer sizes are configurable.

```python
class FCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

model = FCNN(10, [64, 128, 128, 64, 32], 1)
```

✔️ Scalable (change architecture with a list)
✔️ Great for experiments / hyperparameter tuning

---

# 🧩 4. Using `nn.ModuleList`

Gives flexibility while keeping structure explicit.

```python
class FCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Linear(10, 64),
            nn.Linear(64, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 32)
        ])
        
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)

model = FCNN()
```

✔️ More flexible than `Sequential`
✔️ Allows custom logic inside loop

---

# 🧠 5. With Dropout / BatchNorm (Improved FCNN)

Adds regularization and stability.

```python
class FCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)
```

✔️ Better generalization
✔️ Common in real-world models

---

# ⚙️ 6. Functional Style (No predefined layers)

Less common, but useful for experimentation.

```python
class FCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(10, 64)),
            nn.Parameter(torch.randn(64, 128)),
            nn.Parameter(torch.randn(128, 128)),
            nn.Parameter(torch.randn(128, 64)),
            nn.Parameter(torch.randn(64, 32)),
            nn.Parameter(torch.randn(32, 1))
        ])

    def forward(self, x):
        for i in range(len(self.weights) - 1):
            x = torch.relu(x @ self.weights[i])
        return x @ self.weights[-1]
```

✔️ Full manual control
❌ Not recommended for most use cases

---

# 🧾 Summary of Approaches

| Method             | Flexibility | Use Case             |
| ------------------ | ----------- | -------------------- |
| `nn.Sequential`    | Low         | Quick prototypes     |
| Custom `nn.Module` | High        | Standard practice    |
| Loop-based dynamic | Very high   | Research / tuning    |
| `ModuleList`       | High        | Custom forward logic |
| With BN/Dropout    | High        | Production models    |
| Functional/manual  | Extreme     | Experimental         |

---

# 💡 Pro Tips

* Use **ReLU** for most hidden layers
* Initialize weights if needed (`nn.init`)
* Always match input/output sizes carefully
* Use GPU: `model.to(device)`

---

