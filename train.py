import torch
import torch.nn as nn
import wandb
import os

# Данні для прикладу
X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])

# Проста модель
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Ініціалізація wandb
wandb.init(project="linear-regression-pytorch", entity="s-oksana-set-university")

model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Навчання
for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    wandb.log({"loss": loss.item()})

# Збереження моделі локально
os.makedirs("artifacts", exist_ok=True)
model_path = "artifacts/linear_regression_model.pth"
torch.save(model.state_dict(), model_path)

# Завантаження моделі як артефакт
artifact = wandb.Artifact("linear_regression_model", type="model")
artifact.add_file(model_path)
wandb.log_artifact(artifact)

wandb.finish()
print(f"✅ Модель збережена і залита в wandb: {model_path}")
