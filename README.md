# Лінійна регресія з CI/CD

MLOps-процес для навчання та деплою моделі лінійної регресії з використанням: **PyTorch**, **Ray Serve** та **GitHub Actions**.


## CI/CD з GitHub Actions автоматизує:

1. ✅ Отримання коду з репозиторію
2. ✅ Встановлення Python і залежностей
3. ✅ Навчання моделі (train.py)
4. ✅ Деплой з використанням Ray Serve (run_serve.py)

Файл пайплайну розташований тут: .github/workflows/train_deploy.yml


Pipeline запускається:
- при кожному пуші в гілку main
- або вручну через **"Run workflow"**


## Інтеграція з Weights & Biases

Навчання моделі логуються через [Weights & Biases (wandb)](https://wandb.ai/).

Для локального запуску потрібно створити `.env` файл з такими змінними:

```env
WANDB_API_KEY=key #(потрібно вказати власний з https://wandb.ai/)
WANDB_PROJECT=linear-regression-pytorch
WANDB_ENTITY=account #(потрібно вказати власний з https://wandb.ai/)


