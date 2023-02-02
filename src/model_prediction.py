import streamlit as st
from tensorflow.keras.models import load_model
from reading_params import ReadParams
import numpy as np
import time



def model_prediction():
    st.set_page_config(
    page_title="Movie Review Sentiment Analysis App",
    page_icon=":random",
    layout="wide")

    params = ReadParams().read_params()
    model_filepath = params['Model_paths']['model_path']
    loaded_model = load_model(model_filepath)

    col1, col2 = st.columns(2, gap="small")

    with col1:
        st.image('Images/manga.jpg')

    with col2:
        st.title('Movie Review Sentiment Analysis')
        st.header("Please provide your review on your favorite or least favorite superhero movie.")
        review = st.text_area("Movie Review")

        predict_btn = st.button('Predict Sentiment')

        with st.spinner("Please wait..."):
            if review and predict_btn:
                prediction = loaded_model.predict([review])
                prediction = np.squeeze(prediction)

                if prediction <= 0.5:
                    st.error('You have given negative feedback for the film.')
                else:
                    st.success("You have given positive feedback for the film.")


if __name__ == "__main__":
    model_prediction()