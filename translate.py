# Import all of the dependencies
import streamlit as st
import os
import imageio
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model
import requests  # To make the API call for translation

# Set the layout to the streamlit app as wide
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet Full Stack App')
# Generating a list of options or videos
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns
col1, col2 = st.columns(2)

# Function to translate text using Google Translate API
def translate_text(text, target_language, api_key):
    url = "https://translation.googleapis.com/language/translate/v2"
    params = {
        "q": text,
        "target": target_language,
        "key": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()['data']['translations'][0]['translatedText']
    else:
        st.error("Translation failed.")
        return text

if options:
    # Rendering the video
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..', 'data', 's1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)

    with col2:
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif', width=400)

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(f"Prediction: {converted_prediction}")

        # Add multilingual translation
        st.info('Select a language for translation:')
        language_options = {"Hindi": "hi", "Marathi": "mr", "Tamil": "ta", "Gujarati": "gu", "Bengali": "bn", "English": "en"}
        selected_language = st.selectbox("Choose a language", list(language_options.keys()))

        # API Key for Google Translate
        api_key = "YOUR_GOOGLE_API_KEY_HERE"  # Add your API key here

        if st.button("Translate"):
            translated_text = translate_text(converted_prediction, language_options[selected_language], api_key)
            st.text(f"Translated Output ({selected_language}): {translated_text}")
