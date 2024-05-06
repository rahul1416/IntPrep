import streamlit as st
import os
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from score import generate_final_scores
from pathlib import Path
import time
import fire

# import frame_extractor
# import settings

# from st_audiorec import st_audiorec
import cv2
import torch
import pandas as pd
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
from deep_emotion import Deep_Emotion
import pyaudio 
import wave
import speech_recognition as sr
import threading
# import vosk
import wave


# Initialize global dataframe to store processed outputs
if 'global_df' not in st.session_state:
    st.session_state.global_df = pd.DataFrame(columns=['Video', 'Question', 'Emotions', 'Transcribed_Text'])


def extract_audio_from_mp4(mp4_path, audio_path):
    clip = VideoFileClip(mp4_path)
    audio = clip.audio
    audio.write_audiofile(audio_path)

def speech_to_text(audio_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data)
    return text

def detect_emotion(frame):
    model = Deep_Emotion()
    state_dict = torch.load("model/emotion_model.pth")
    model.load_state_dict(state_dict)
    emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    emotion_label = None
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face_pil = Image.fromarray(face)
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        face_tensor = transform(face_pil).unsqueeze(0)
        with torch.no_grad():
            outputs = model(face_tensor)
            _, predicted = torch.max(outputs, 1)
            emotion_label = emotion_labels[predicted.item()]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    if emotion_label is None:
        emotion_label = "No_face_detected"

    return frame, emotion_label

    



def main():
    st.title("IntPrep: Your Interview Preparation Tool")

    question_number = st.selectbox("Select Question Number", options=["Question 1", "Question 2", "Question 3", "Question 4", "Question 5", "Question 6", "Question 7"])

    # Display selected question
    if question_number == "Question 1":
        question_text = "Can you explain your experience in building and fine-tuning chatbots and NLP systems, specifically within the context of the travel domain?"
    elif question_number == "Question 2":
        question_text = "How familiar are you with transformer models such as OpenAI and Hugging Face, and can you discuss any projects where you utilized these models?"
    elif question_number == "Question 3":
        question_text = "In your opinion, what are the key challenges in architecting and designing a chatbot for corporate travel, and how would you approach addressing them?"
    elif question_number == "Question 4":
        question_text = "Describe your approach to curating, testing, and maintaining datasets for travel chat conversations. How do you ensure data quality and relevance?"
    elif question_number == "Question 5":
        question_text = "Can you discuss a time when you had to determine intents and create maintainable chat workflows for a complex system? What were some considerations you had to take into account?"
    elif question_number == "Question 6":
        question_text = "How do you ensure scalability and performance when implementing database models for a platform that serves multiple customers with varying needs?"
    elif question_number == "Question 7":
        question_text = "Have you worked on integrating AI-driven platforms into various communication channels such as email, Slack, and SMS? If so, what were some challenges you encountered, and how did you overcome them?"

    st.write(question_text)

    # Display file uploader widget
    uploaded_file = st.file_uploader("Upload MP4 file", type=["mp4"])

    if uploaded_file is not None:
        # Display video
        st.video(uploaded_file)

        # Save the file as "clip.mp4" in the main directory
        save_path = "clip.mp4"
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())

        # Call the function to extract audio from "clip.mp4" and save it as "clip_audio.wav"
        extract_audio_from_mp4("clip.mp4", "clip_audio.wav")

        # Transcribe audio to text
        text = speech_to_text("clip_audio.wav")

        # Initialize emotions list
        emotions_list = []

        video_path = "clip.mp4"
        cap = cv2.VideoCapture(video_path)

        # Check if the video file was successfully opened
        if not cap.isOpened():
            st.error(f"Error opening video file: {video_path}")
            return
        else:
            st.success(f"Video file {video_path} was successfully opened.")

            # Get the frames per second (fps) of the video
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Process each frame in the video
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                # Apply your custom function to the frame (e.g., display the frame)
                # Replace this with your actual processing function
                # For example, you can apply filters, resize, or analyze the frame
                frame, emotion_label = detect_emotion(frame)
                emotions_list.append(emotion_label)

                # Press 'q' to exit the loop
                if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
                    break

        
            cap.release()
            cv2.destroyAllWindows()

        
        st.session_state.global_df = pd.concat([
            st.session_state.global_df, 
            pd.DataFrame(
                {'Video': [uploaded_file.name], 
                 'Question': [question_text], 
                 'Emotions': [emotions_list], 
                 'Transcribed_Text': [text]}
            )
        ], ignore_index=True)

        # Display processed outputs
        st.subheader("Processed Outputs")
        st.write(st.session_state.global_df)
        print(st.session_state.global_df)

        emotion_scores = {
            'Angry': -2,
            'Disgust': -1,
            'Fear': 2,
            'Happy': 2,
            'Neutral': 1,
            'Sad': -1,
            'Surprise': 0.5,
            'No_face_detected': 0
        }
        if st.button("Submit"):
            score = generate_final_scores(st.session_state.global_df, emotion_scores)
            st.success(f"Your score is: {score}")

if __name__ == "__main__":
    main()