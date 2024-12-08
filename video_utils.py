import cv2
import base64
from typing import List
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
from langsmith import traceable
load_dotenv()
client = OpenAI()

@traceable
def extract_frames(video_path: str, frame_rate: int, target_size: int = 512) -> List[str]:
    """Extract frames from the video, resize to 512x512, and return base64-encoded strings."""
    cap = cv2.VideoCapture(video_path)
    frames_base64 = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Resize frame to (512x512)
            resized_frame = cv2.resize(frame, (target_size, target_size))

            # Encode as JPEG and convert to base64
            _, buffer = cv2.imencode('.jpg', resized_frame)
            frame_base64 = base64.b64encode(buffer).decode("utf-8")
            frames_base64.append(frame_base64)

        frame_count += 1

    cap.release()
    return frames_base64


def transcribe(video_path: str) -> str:
    try:
        audio_file = open(video_path, "rb")  # Assuming the audio file is extracted from the video
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcription.text
    except Exception as e:
        print(e)
        return ""
