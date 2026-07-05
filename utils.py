import cv2
import numpy as np

def load_image(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    nparr = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def load_video(uploaded_file):
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    return cv2.VideoCapture("temp_video.mp4")