# Классификатор отзывов (Reviews Classifier)

Проект по обучению и развертыванию модели **intfloat/multilingual-e5-small** для поиска негативных отзывов с использованием **NVIDIA Triton Inference Server**.

## Цель проекта

Автоматическая классификация текстовых отзывов для выявления негативных обращений.  
Задача решается как **бинарная классификация**:
- **0** — позитивный отзыв (Positive)
- **1** — негативный отзыв (Negative)

---

## Структура проекта
```text
reviews-classifier/
├── docker-compose.yaml
├── README.md
├── triton_rc/
│   └── models/
│       ├── bert_classifier/
│       │   ├── 1/
│       │   │   └── model.onnx
│       │   └── config.pbtxt
│       ├── text_tokenizer/
│       │   ├── 1/
│       │   │   └── model.py
│       │   └── config.pbtxt
│       └── text_classifier_ensemble/
│           ├── 1/
│           └── config.pbtxt
├── streamlit/
│   └── app.py
└── notebooks/
    ├── test_triton.ipynb
    └── train_notebook.ipynb
```
Описание:
---
| Путь | Описание |
|------|----------|
| `docker-compose.yaml` | Docker Compose для Triton + Streamlit |
| `triton_rc/models/bert_classifier/` | ONNX модель классификатора |
| `triton_rc/models/text_tokenizer/` | Python бэкенд для токенизации |
| `triton_rc/models/text_classifier_ensemble/` | Ensemble пайплайн |
| `streamlit/app.py` | Веб-интерфейс для инференса |
| `notebooks/train_notebook.ipynb` | Обучение и экспорт модели |
| `notebooks/test_triton.ipynb` | Тестирование Triton |

## Особенности

- **Fine-tuning** модели E5 для бинарной классификации
- Предобработка и анализ датасета `geo-reviews-dataset-2023.csv`
- Экспорт обученной модели в **ONNX** формат
- Развертывание **Triton Inference Server** через Docker Compose
- **Streamlit** клиент для инференса

---

## Запуск
### 1. Скачать обученную модель

Модель в формате ONNX доступна для скачивания по ссылке:  
[Ссылка на Google Drive](https://drive.google.com/file/d/1ZC3gq70A4hYd0gYseF96TUm2_z77gamJ/view?usp=sharing)

Скачайте файл `model.onnx` и поместите его в директорию:
- triton_rc/models/bert_classifier/1/model.onnx

### 2. Запустить контейнеры

Запуск тритона с обученной моделью и streamlit интерфейса:
```bash
docker-compose up -d --build
```

Будут запущены два сервиса:
- triton_rc — Triton Inference Server (порты: `8000`, `8001`, `8002`)
- streamlit_app — веб-интерфейс (порт: `8501`)

Веб интерфес будет по адресу: http://localhost:8501
