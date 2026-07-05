import numpy as np
import cv2
import os

MODEL_PATH = "models/mesonet.tflite"

class DeepfakeDetector:
    def __init__(self):
        self.model = None
        self.use_placeholder = True
        if os.path.exists(MODEL_PATH):
            try:
                import tensorflow as tf
                self.model = tf.lite.Interpreter(model_path=MODEL_PATH)
                self.model.allocate_tensors()
                self.use_placeholder = False
            except:
                pass
        if self.use_placeholder:
            print("INFO: No TFLite model found – using blur heuristic for demo.")

    def preprocess_frame(self, frame):
        resized = cv2.resize(frame, (224, 224))
        normalized = resized / 255.0
        return np.expand_dims(normalized, axis=0).astype(np.float32)

    def predict(self, frame):
        if self.use_placeholder:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            fake_prob = 1.0 - min(laplacian_var / 100.0, 1.0)
            confidence = 0.6 + 0.3 * (1.0 - fake_prob)
            return float(fake_prob), float(confidence)
        else:
            input_tensor = self.preprocess_frame(frame)
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()
            self.model.set_tensor(input_details[0]['index'], input_tensor)
            self.model.invoke()
            output = self.model.get_tensor(output_details[0]['index'])
            fake_prob = output[0][0]
            confidence = 0.9
            return float(fake_prob), float(confidence)

detector = DeepfakeDetector()

def analyze_frame(frame):
    fake_prob, conf = detector.predict(frame)
    return {"fake_probability": fake_prob, "confidence": conf}