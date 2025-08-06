import torch

# Relationship: f = w * x
# Our dataset follows f = 2 * x

# Training data
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)    # Inputs
y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)    # Outputs

# Initial weight (parameter to learn)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# Forward pass function
def forward(x):
    return w * x

# Loss function (Mean Squared Error)
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()

# Initial prediction check
print(f'Prediction before training: {forward(torch.tensor(5.0)).item():.3f}')

# Hyperparameters
learning_rate = 0.01
n_iters = 1000

# Training loop
for epoch in range(n_iters):
    # 1. Forward pass (prediction)
    y_pred = forward(x)
    
    # 2. Compute loss
    l = loss(y, y_pred)
    
    # 3. Backward pass (compute gradient)
    l.backward()
    dw = w.grad.item()
    
    # 4. Update weight
    with torch.no_grad():
        w -= learning_rate * dw
    
    # 5. Zero the gradients for the next iteration
    w.grad.zero_()
    
    # 6. Print progress every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch + 1}/{n_iters}, Loss: {l.item():.4f}, w: {w.item():.4f}')

# Final prediction check
print(f'Prediction after training: {forward(torch.tensor(5.0)).item():.3f}')
