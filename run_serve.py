import os
from dotenv import load_dotenv
import ray
from ray import serve
from model_def import entrypoint

# Завантажуємо змінні середовища з .env
load_dotenv()

# Ініціалізація локального Ray кластеру
ray.init(
    runtime_env={
        "working_dir": ".",  # поточна директорія з усіма файлами
        "pip": [
            "wandb",
            "python-dotenv",
            "torch",
            "fastapi",
            "uvicorn",
            "pydantic",
            "requests"
        ],
        "env_vars": {
            "WANDB_PROJECT": os.getenv("WANDB_PROJECT", "linear-regression-pytorch"),
            "WANDB_ENTITY": os.getenv("WANDB_ENTITY", "s-oksana-set-university"),
            "WANDB_MODEL_ARTIFACT": os.getenv(
                "WANDB_MODEL_ARTIFACT",
                "s-oksana-set-university/linear-regression-pytorch/linear_regression_model:v0"
            ),
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
            "WANDB_MODE": "online",
            "WANDB_SILENT": "true"
        }
    }
)

# Запуск моделі через Ray Serve
serve.run(entrypoint, name="linear-regression")

print("✅ Модель успішно задеплоєна! Доступна за адресою: http://127.0.0.1:8000/")
