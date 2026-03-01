# Dushanbe School ML Analysis

Короткий проект по анализу учебных данных и сравнению моделей для предсказания `Рез_экзамена`.

## Что внутри
- `Data_making/` — подготовка и синтетический датасет.
- `EDA/EDA.ipynb` — анализ распределений, связей и обоснование новых фичей.
- `Models/Baseline.ipynb` — baseline (LogReg) без утечки.
- `Models/Catboost.ipynb` — CatBoost с финальными фичами, подбором threshold и SHAP.
- `Models/Compare models/compare_models.ipynb` — сравнение метрик моделей.

## Быстрый старт
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Дальше запускайте ноутбуки в порядке: `EDA -> Baseline -> Catboost -> compare_models`.

