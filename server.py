import streamlit as st
from fastai.vision.all import *

def extract_last_folder_name(path):
   parts = str(path).split("/")
   return parts[5]

model_path = "food_prediction_model.pkl"
food_predict_model = load_learner(model_path)


def predict_food(image):
   img = PILImage.create(image)  # Use PILImage.create to open the image


   resized = img.resize((224, 224))


   prediction = food_predict_model.predict(resized)


   breed_index = prediction[1].item()


   accuracy = prediction[2][breed_index]




   img.show(title=f"{prediction[0]} - {(accuracy * 100):.2f}%")


sample_dir = "/kaggle/input/fruit-and-vegetable-image-recognition/test"


sample_files = get_image_files(sample_dir)


# for i in range(10):
   # predict_food(sample_files[i])


st.text("Food Prediction Model")
st.text("Built by Jayden Hang")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])


if uploaded_file is not None:
   prediction = predict(uploaded_file)
   st.image(uploaded_file, caption=prediction, use_column_width=True)