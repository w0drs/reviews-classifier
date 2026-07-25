import streamlit as st
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

# Настройка страницы
st.set_page_config(
    page_title="Reviews Classifier",
    page_icon="🤖",
    layout="centered"
)

st.title("Классификатор отзывов")
st.markdown("Введите текст для анализа")


# Подключение к Triton через gRPC
def get_client():
    try:
        client = grpcclient.InferenceServerClient(url="triton:8001")
        if client.is_server_live():
            return client
        else:
            st.error("Сервер не отвечает")
            return None
    except Exception as e:
        st.error(f"Ошибка подключения: {e}")
        return None


# Создаем клиент в session_state
if 'client' not in st.session_state:
    st.session_state.client = get_client()

client = st.session_state.client

if client is None:
    st.stop()

try:
    client.is_server_live()
    st.success("Подключение к Triton серверу установлено")
except Exception as e:
    st.error(f"Не удалось подключиться к Triton серверу: {e}")
    st.session_state.client = None
    st.stop()

try:
    if client.is_model_ready("text_classifier_ensemble"):
        st.success("Модель text_classifier_ensemble готова")
    else:
        st.error("Модель text_classifier_ensemble не готова")
        st.stop()
except Exception as e:
    st.error(f"Ошибка проверки модели: {e}")
    st.stop()

# Поле ввода текста
text = st.text_area(
    "Введите текст для классификации:",
    placeholder="Например: Это место замечательно!",
    height=100
)

if st.button("Классифицировать", type="primary"):
    if not text.strip():
        st.warning("Пожалуйста, введите текст")
        st.stop()

    with st.spinner("Классифицируем отзыв..."):
        try:
            input_tensor = grpcclient.InferInput(
                name="TEXT",
                shape=[1, 1],
                datatype="BYTES"
            )
            input_tensor.set_data_from_numpy(
                np.array([[text.encode("utf-8")]], dtype=np.object_)
            )

            response = client.infer(
                model_name="text_classifier_ensemble",
                inputs=[input_tensor]
            )

            output = response.as_numpy("OUTPUT")
            logit = float(output[0][0])
            prob_negative = 1 / (1 + np.exp(-logit))
            prob_positive = 1 - prob_negative

            if prob_negative >= 0.5:
                label = "Negative"
                delta_text = "Отрицательный"
                delta_color = "inverse"
            else:
                label = "Positive"
                delta_text = "Положительный"
                delta_color = "normal"

            st.divider()
            st.subheader("Результат классификации")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Вероятность отрицательного",
                    value=f"{prob_negative:.2%}"
                )
                st.metric(
                    label="Вероятность положительного",
                    value=f"{prob_positive:.2%}"
                )

            with col2:
                st.metric(
                    label="Класс",
                    value=label,
                    delta=delta_text,
                    delta_color=delta_color
                )

            st.progress(prob_negative, text=f"Уверенность в отрицательном: {prob_negative:.2%}")

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

    **Классы:**
    - 0 = Positive (положительный)
    - 1 = Negative (отрицательный)

    **Как это работает:**
    1. Текст токенизируется
    2. Подается в классификатор
    3. Выдается вероятность отрицательного класса
    """)

    st.divider()
    st.subheader("Статус моделей")
    try:
        models = client.get_model_repository_index()
        for model in models:
            if model['name'] in ['text_tokenizer', 'bert_classifier', 'text_classifier_ensemble']:
                st.success(f"{model['name']}")
    except Exception as e:
        st.warning(f"Не удалось получить статус: {e}")