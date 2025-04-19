import streamlit as st
import tempfile
import requests
import os

from transcriber import transcribe_audio
from text_analysis import analyze_text
from utils import clean_text, generate_kpis, save_results

st.title("\U0001F3A4 Speech-to-Text Characterization")

input_option = st.radio("Choose input method:", ("Upload File", "Paste URL"))

audio_path = None

if input_option == "Upload File":
    uploaded_file = st.file_uploader("Upload an audio file (MP3/WAV)", type=["mp3", "wav"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(uploaded_file.read())
            audio_path = tmp.name

elif input_option == "Paste URL":
    url = st.text_input("Paste direct audio URL or YouTube link")
    if url:
        if "youtube.com" in url or "youtu.be" in url:
            import yt_dlp
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': 'temp_audio.%(ext)s',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                audio_path = 'temp_audio.mp3'
        else:
            response = requests.get(url)
            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(response.content)
                    audio_path = tmp.name
            else:
                st.error("Failed to fetch audio from URL")

if audio_path:
    text, duration, rtf = transcribe_audio(audio_path)

    st.subheader("Transcribed Text")
    st.write(text)

    cleaned_text = clean_text(text)
    sentiment, emotion, topic = analyze_text(cleaned_text)
    kpis = generate_kpis(text, duration, rtf, sentiment, emotion, topic)

    st.subheader("Characterization Summary")
    for k, v in kpis.items():
        st.write(f"**{k}**: {v}")

    st.subheader("Topic Scores")
    for label, score in zip(topic['labels'], topic['scores']):
        st.write(f"{label}: {round(score, 2)}")