''' Streamlit application for ASR

- The app uses wav2vec2 pretrained and fine-tuned model for the inference data.
- You can record the audio and upload the .wav files to generate transcription.

# IMPORTANT:
# please run the script and then type 'streamlit run asr_streamlit_app.py'
# in the terminal to run the streamlit app.

'''

## Install dependencies
import os
# os.system('pip install datasets')
# os.system('pip install transformers')
# os.system('pip install torchaudio')
# os.system('pip install librosa')
# os.system('pip install jiwer')
# os.system('pip install soundfile')
# os.system('pip install sounddevice')
# os.system('pip install wavio')
# os.system('pip install glob2')

from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from datasets import load_dataset
import soundfile as sf
import torch
import torchaudio
import librosa
import sounddevice as sd
import wavio
import streamlit as st
import glob
import os


# functions to read audio files, record audio, and save the recorded audio.
def read_audio(file):
    with open(file, "rb") as audio_file:
        audio_bytes = audio_file.read()
    return audio_bytes

def record(duration=5, fs=48000):
    sd.default.samplerate = fs
    sd.default.channels = 1
    myrecording = sd.rec(int(duration * fs))
    sd.wait(duration)
    return myrecording

def save_record(path_myrecording, myrecording, fs):
    wavio.write(path_myrecording, myrecording, fs, sampwidth=2)
    return None


# calling the pretrained model.
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Streamlit app
st.header("1. Record your own voice")

filename = st.text_input("Choose a filename: ")

if st.button(f"Click to Record"):
    if filename == "":
        st.warning("Choose a filename.")
    else:
        record_state = st.text("Recording...")
        duration = 8  # seconds
        fs = 48000
        myrecording = record(duration, fs)
        record_state.text(f"Saving sample as {filename}.wav")

        path_myrecording = f"./{filename}.wav"

        save_record(path_myrecording, myrecording, fs)
        record_state.text(f"Done! Saved sample as {filename}.wav")

        st.audio(read_audio(path_myrecording))

        audio, rate = librosa.load(path_myrecording, sr=16000)
        input_values = tokenizer(audio, return_tensors="pt").input_values

        # Storing logits (non-normalized prediction values)
        logits = model(input_values).logits

        # Storing predicted ids
        prediction = torch.argmax(logits, dim=-1)

        # Passing the prediction to the tokenzer decode to get the transcription
        transcription = tokenizer.batch_decode(prediction)[0]

        st.write(f'Transcription =', transcription)


"## 2. Choose an audio record"

audio_folder = "."
filenames = glob.glob(os.path.join(audio_folder, "*.wav"))
selected_filename = st.selectbox("Select a file", filenames)

if selected_filename is not None:

    # Reading the audio file.
    audio_file = open(selected_filename, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')

    # Sample rate is set to 16KHz to match with the model inputs
    audio, rate = librosa.load(selected_filename, sr=16000)
    input_values = tokenizer(audio, return_tensors="pt").input_values

    # Storing logits (non-normalized prediction values)
    logits = model(input_values).logits

    # Storing predicted ids
    prediction = torch.argmax(logits, dim=-1)

    transcription = tokenizer.batch_decode(prediction)[0]

    st.write(f'Transcription =', transcription)


# IMPORTANT:
# please run the script and then type 'streamlit run asr_streamlit_app.py'
# in the terminal to run the streamlit app.
