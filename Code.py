import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image, ImageOps

# -------------------------
# 1. Build CNN Model
# -------------------------
def build_model(num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# -------------------------
# 2. Load Pretrained Model (MNIST for digits 0–9)
# -------------------------
@st.cache_resource
def load_trained_model():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train.reshape((-1,28,28,1))
    x_test = x_test.reshape((-1,28,28,1))

    model = build_model(10)
    model.fit(x_train, y_train, epochs=2, batch_size=128, verbose=0)  # quick train
    return model

model = load_trained_model()

# -------------------------
# 3. Streamlit UI
# -------------------------
st.set_page_config(page_title="CNN Shape & Number Trainer", page_icon="🎮")
st.title("🎮 CNN Shape & Number Trainer Game")

st.write("👉 Draw a **number (0–9)** on the canvas or upload an image, and the AI will guess it!")

# Upload Image
uploaded_file = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # grayscale
    image = ImageOps.invert(image)  # invert to match MNIST
    image = image.resize((28,28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1,28,28,1)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.image(image, caption="Your Drawing (Preprocessed)", width=150)
    st.success(f"✅ I think you drew: **{predicted_class}**")

# Drawing Pad (using Streamlit-Draw-Canvas)
from streamlit_drawable_canvas import st_canvas

st.subheader("✏️ Draw a digit here:")
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=200,
    height=200,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = cv2.resize(cv2.cvtColor(canvas_result.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY), (28,28))
    img = img / 255.0
    img = img.reshape(1,28,28,1)

    if st.button("🎯 Predict My Drawing"):
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        st.success(f"🎉 The model thinks you drew: **{predicted_class}**")
