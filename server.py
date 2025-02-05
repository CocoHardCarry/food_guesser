import streamlit as st
from fastai.vision.all import *


def extract_last_folder_name(path):
    parts = str(path).split("/")
    return parts[6]

model_path = Path("food_prediction_model.pkl")
food_predict_model = load_learner(model_path)

def predict_food(image):
    img = PILImage.create(image)

    resized = img.resize((224, 224))

    prediction = food_predict_model.predict(resized)

    predicted_class = prediction[0]
    breed_index = prediction[1].item()
    accuracy = prediction[2][breed_index]

    if accuracy > 0.9:
        return f"{prediction_class} - {accuracy * 100}% confident."
    else:
        return f"I am not sure what this is, it might be {predicted_class} - {accuracy * 100}% confident."

# Streamlit app UI
st.title("Fruit/Vegetable Prediction Model")
st.text("Built by Jayden Hang")

# File uploader for user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# If a file is uploaded, make a prediction
if uploaded_file is not None:
    prediction_class = predict_food(uploaded_file)
    st.image(uploaded_file, caption=prediction_class, use_column_width=True)
