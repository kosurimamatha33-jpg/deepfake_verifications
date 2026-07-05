import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def get_face_and_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    return (x, y, w, h), [(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes]

def eye_aspect_ratio_from_bbox(eye_bbox):
    (ex, ey, ew, eh) = eye_bbox
    if eh == 0:
        return 0.5
    return ew / (2.0 * eh)

def detect_blink_haar(frame, prev_ear=None):
    face, eyes = get_face_and_eyes(frame)
    if face is None or len(eyes) < 2:
        return False, None
    eyes_sorted = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
    ears = [eye_aspect_ratio_from_bbox(e) for e in eyes_sorted]
    avg_ear = np.mean(ears) if ears else 0.3
    if prev_ear is not None and avg_ear < prev_ear * 0.6:
        return True, avg_ear
    return False, avg_ear

def detect_head_movement_haar(prev_face, face):
    if prev_face is None or face is None:
        return False
    (x1, y1, w1, h1) = prev_face
    (x2, y2, w2, h2) = face
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return (dx + dy) > 20

def liveness_check(frame, prev_face=None, prev_ear=None):
    face, eyes = get_face_and_eyes(frame)
    if face is None:
        return {"liveness": False, "reason": "No face detected"}, None, None
    blink, ear = detect_blink_haar(frame, prev_ear)
    head_moved = detect_head_movement_haar(prev_face, face)
    liveness = blink or head_moved
    reason = []
    if blink:
        reason.append("Blink detected")
    if head_moved:
        reason.append("Head movement detected")
    if not liveness:
        reason.append("No blink or movement - possible replay attack")
    return {"liveness": liveness, "reason": " | ".join(reason)}, face, ear