# Прогнозирование открытия депозитов в банке

| | |
|---|---|
| ![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white) | ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7.2-F7931E?logo=scikit-learn&logoColor=white) |
| ![Pandas](https://img.shields.io/badge/Pandas-2.3.1-150458?logo=pandas&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white) |
| ![License](https://img.shields.io/badge/License-MIT-yellow.svg) | ![Status](https://img.shields.io/badge/Status-Completed-success.svg) |

## Обзор проекта

**Бизнес-задача:** повысить эффективность маркетинговых кампаний банка за счёт прогноза клиентов, склонных открыть срочный депозит.

**Техническая задача:** построить и сравнить модели бинарной классификации, подобрать гиперпараметры и описать ключевые факторы отклика.

**Данные:** [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) (португальский банк, телефонные кампании). В репозитории — подготовленный файл `data/bank_fin.csv` (формат с `;`, поле `balance` в текстовом виде).

## Структура проекта

```text
bank-term-deposit-prediction/
├── data/
│   └── bank_fin.csv              # 11 162 строк (исходная выгрузка)
├── notebooks/
│   └── bank_deposit_classification.ipynb
├── LICENSE
├── README.md
├── requirements.txt
└── .gitignore
```

## Описание данных

### Целевая переменная

- **`deposit`**: открыл ли клиент депозит (`yes` / `no`)

### Основные признаки (английские имена в CSV)

| Группа | Поля |
|--------|------|
| Клиент | `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan` |
| Текущий контакт | `contact`, `month`, `day`, `duration` |
| Кампания | `campaign`, `pdays`, `previous`, `poutcome` |

### Статистика (по ноутбуку)

| Этап | Записей | Комментарий |
|------|---------|-------------|
| Загрузка | 11 162 | исходный `bank_fin.csv` |
| После IQR по `balance` | 10 105 | удалено 1 057 выбросов |
| Целевая (до split) | 46.3% `yes`, 53.7% `no` | относительно сбалансировано |
| Пропуски | 25 в `balance` | заполнение медианой |
| Признаки после encoding | ~45 числовых + OHE | SelectKBest → 15 для моделей |

### Источник и лицензия данных

- Датасет: UCI Machine Learning Repository — *Bank Marketing* (Moro et al., 2014).
- Публикация: [DOI 10.24432/C5K306](https://doi.org/10.24432/C5K306).
- Использование в учебных/исследовательских целях; при распространении соблюдайте условия UCI и указывайте источник.

### Важно: признак `duration`

`duration` — длительность **уже состоявшегося** звонка. Для скоринга **до** контакта этот признак недоступен (типичный data leakage в задаче Bank Marketing). В проекте он используется для анализа и максимизации метрик на исторических данных; для продакшен-сценария «до звонка» нужна отдельная модель без `duration`.

## Pipeline (ноутбук)

1. Загрузка и парсинг `balance`, обработка пропусков, IQR-фильтрация выбросов.
2. EDA: распределения, корреляции, VIF (`statsmodels`), сезонность по `month`.
3. Feature engineering: возрастные группы, `deposit` → бинарная метка, LabelEncoder + One-Hot.
4. `train_test_split` (33% test, `stratify`, `random_state=42`), MinMaxScaler, SelectKBest (`k=15`).
5. Модели: логистическая регрессия, дерево (GridSearchCV, `cv=5`), случайный лес, градиентный бустинг, stacking (`StackingClassifier`, `cv=5`).
6. Optuna для случайного леса: 50 trials, в objective — `cross_val_score` с `cv=3`, метрика F1.

## Результаты на тестовой выборке

Метрики из выполненного ноутбука (`random_state=42`):

| Модель | Accuracy | Precision | Recall | F1 | Параметры |
|--------|----------|-----------|--------|-----|-----------|
| Логистическая регрессия | 0.80 | 0.83 | 0.73 | 0.78 | `solver='sag'`, `max_iter=1000` |
| Решающее дерево | 0.80 | 0.83 | 0.73 | 0.78 | `max_depth=7`, `criterion='entropy'` |
| **Случайный лес** | **0.83** | **0.81** | **0.83** | **0.82** | `n_estimators=100`, `max_depth=10` |
| Градиентный бустинг | 0.82 | 0.80 | 0.83 | 0.81 | `learning_rate=0.05`, `n_estimators=300` |
| Stacking | 0.82 | 0.81 | 0.81 | 0.81 | RF + DT + GB → логистическая мета-модель |
| Случайный лес (Optuna) | 0.83 | — | — | 0.82 | `n_estimators=182`, `max_depth=10`, `min_samples_leaf=2` |

**Лучшая модель по F1 на hold-out:** случайный лес (базовый и Optuna ~0.82–0.83 accuracy).

```python
# Optuna — лучшие параметры
{
    "n_estimators": 182,
    "max_depth": 10,
    "min_samples_leaf": 2,
    "criterion": "gini",
}
```

### Выводы из EDA (подтверждены в ноутбуке)

- Сильнейшая линейная связь с целевой: **`duration`** (корреляция ≈ +0.46).
- Также заметны `poutcome`, `balance`, категории контакта и месяца кампании.
- В случайном лесe наибольшая **feature importance** — у `duration` (см. график в ноутбуке).

## Быстрый старт

```bash
git clone https://github.com/theKerimKerimov/bank-term-deposit-prediction.git
cd bank-term-deposit-prediction

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
jupyter lab notebooks/bank_deposit_classification.ipynb
```

Выполняйте ячейки сверху вниз. Путь к данным из ноутбука: `../data/bank_fin.csv`, разделитель `;`.

## Технологический стек

- Python 3.10+
- pandas, numpy, scikit-learn, scipy
- matplotlib, seaborn, plotly (визуализации Optuna)
- optuna, statsmodels
- JupyterLab / Notebook

Зависимости с версиями — в `requirements.txt`.

## Автор

Karim — 2026

- GitHub: [@theKerimKerimov](https://github.com/theKerimKerimov)
- Kaggle: [@kerimkerimov](https://www.kaggle.com/kerimkerimov)
