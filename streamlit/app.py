import streamlit as st
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

# Настройка страницы
st.set_page_config(
    page_title="Reviews Classifier",
    page_icon="🤖",
    layout="centered"
)

st.title("Классификатор отзывов")
st.markdown("Введите текст для анализа")


# Подключение к Triton
@st.cache_resource
def get_client():
    return httpclient.InferenceServerClient(url="localhost:8000")


client = get_client()

# Проверка доступности сервера
try:
    client.is_server_live()
    st.success("Подключение к Triton серверу установлено")
except:
    st.error("Не удалось подключиться к Triton серверу (localhost:8000)")
    st.stop()

# Поле ввода текста
text = st.text_area(
    "Введите текст для классификации:",
    placeholder="Например: Это место замечательно!",
    height=100
)

# Кнопка для отправки
if st.button("Классифицировать", type="primary"):
    if not text.strip():
        st.warning("Пожалуйста, введите текст")
        st.stop()

    # Показываем спиннер
    with st.spinner("Классифицируем отзыв..."):
        try:
            # Подготовка входных данных для Triton
            # Создаем тензор для текста
            input_tensor = httpclient.InferInput(
                name="TEXT",
                shape=[1, 1],  # batch_size=1, 1 текст
                datatype="BYTES"
            )
            # Отправляем текст в байтовом формате
            input_tensor.set_data_from_numpy(
                np.array([[text.encode("utf-8")]], dtype=np.object_)
            )

            # Отправляем запрос к ensemble модели
            response = client.infer(
                model_name="text_classifier_ensemble",
                inputs=[input_tensor]
            )

            # 3. Получаем результат
            output = response.as_numpy("OUTPUT")
            score = float(output[0][0])  # [batch, 1]
            score = 1 / (1 + np.exp(-score))  # сигмоида для преобразования в вероятность

            # Отображение результата
            st.divider()
            st.subheader("Результат классификации")

            # Создаем две колонки
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Сырой logit",
                    value=f"{score:.4f}"
                )

            with col2:
                # Классификация (порог 0.5)
                label = "Positive" if score >= 0.5 else "Negative"
                st.metric(
                    label="Класс",
                    value=label,
                    delta="Положительный" if score >= 0.5 else "Отрицательный",
                    delta_color="normal" if score >= 0.5 else "inverse"
                )

            # Прогресс-бар для наглядности
            st.progress(score, text=f"Уверенность: {score:.2%}")

        except InferenceServerException as e:
            st.error(f"Ошибка сервера: {e}")
        except Exception as e:
            st.error(f"Ошибка: {e}")

# Информация в сайдбаре
with st.sidebar:
    st.header("Информация")
    st.markdown("""
    **Модель:** multilingual-e5-small  
    **Сервер:** Triton Inference Server  
    **Формат:** ONNX  

    **Как это работает:**
    1. Текст токенизируется
    2. Подается в BERT-классификатор
    3. Выдается вероятность положительного класса
    """)

    # Статус моделей
    st.divider()
    st.subheader("Статус моделей")
    try:
        models = client.get_model_repository_index()
        for model in models:
            if model['name'] in ['text_tokenizer', 'bert_classifier', 'text_classifier_ensemble']:
                status = client.get_model_config(model['name'])
                st.success(f"{model['name']}: {status['config']['name']}")
    except:
        st.warning("Не удалось получить статус моделей")