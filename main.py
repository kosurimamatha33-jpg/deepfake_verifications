import sqlite3
import hashlib
import os
import uuid
import json
import secrets
import shutil
import base64
from fastapi import FastAPI, Request, Response, HTTPException, Depends, status, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from typing import Optional, List, Dict
import cv2
import numpy as np

# Import our detection modules
from liveness import liveness_check
from eyebrow import detect_eyebrow_anomaly, reset_frame_history
from deepfake import analyze_frame
from image_analysis import full_image_analysis

DB_PATH = "auth_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            type TEXT NOT NULL,
            file_name TEXT NOT NULL,
            verdict TEXT NOT NULL,
            confidence REAL NOT NULL,
            details_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

# Initialize DB on import
init_db()

app = FastAPI(title="Deepfake Verification API")

# Password utility functions
def hash_password(password: str, salt: str = None) -> tuple:
    if salt is None:
        salt = secrets.token_hex(16)
    pwd_bytes = password.encode('utf-8')
    salt_bytes = salt.encode('utf-8')
    h = hashlib.pbkdf2_hmac('sha256', pwd_bytes, salt_bytes, 100000)
    return h.hex(), salt

# Session utilities
def create_session(user_id: int) -> str:
    session_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO sessions (session_id, user_id) VALUES (?, ?)", (session_id, user_id))
    conn.commit()
    conn.close()
    return session_id

def get_user_from_session(session_id: str) -> Optional[dict]:
    if not session_id:
        return None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.id, u.name, u.email, u.created_at 
        FROM sessions s 
        JOIN users u ON s.user_id = u.id 
        WHERE s.session_id = ?
    """, (session_id,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return dict(user)
    return None

def delete_session(session_id: str):
    if not session_id:
        return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

# Dependency for routes that require authentication
async def get_current_user(request: Request):
    session_id = request.cookies.get("session_id")
    user = get_user_from_session(session_id)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user

# ─────────────────────────────────────────────────────────────
# AUTHENTICATION API
# ─────────────────────────────────────────────────────────────

@app.post("/api/auth/signup")
async def signup(name: str = Form(...), email: str = Form(...), password: str = Form(...)):
    email = email.lower().strip()
    password_hash, salt = hash_password(password)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (name, email, password_hash, salt) VALUES (?, ?, ?, ?)",
            (name, email, password_hash, salt)
        )
        conn.commit()
        user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="Email already registered")
    conn.close()
    
    session_id = create_session(user_id)
    response = JSONResponse(content={"status": "success", "user": {"id": user_id, "name": name, "email": email}})
    response.set_cookie(key="session_id", value=session_id, httponly=True, max_age=86400 * 7, samesite="lax")
    return response

@app.post("/api/auth/signin")
async def signin(email: str = Form(...), password: str = Form(...)):
    email = email.lower().strip()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        raise HTTPException(status_code=400, detail="Invalid email or password")
        
    db_hash = user["password_hash"]
    db_salt = user["salt"]
    
    check_hash, _ = hash_password(password, db_salt)
    if check_hash != db_hash:
        raise HTTPException(status_code=400, detail="Invalid email or password")
        
    session_id = create_session(user["id"])
    response = JSONResponse(content={"status": "success", "user": {"id": user["id"], "name": user["name"], "email": user["email"]}})
    response.set_cookie(key="session_id", value=session_id, httponly=True, max_age=86400 * 7, samesite="lax")
    return response

@app.post("/api/auth/signout")
async def signout(request: Request, response: Response):
    session_id = request.cookies.get("session_id")
    if session_id:
        delete_session(session_id)
    response.delete_cookie(key="session_id")
    return {"status": "success", "message": "Logged out successfully"}

@app.get("/api/auth/me")
async def me(current_user: dict = Depends(get_current_user)):
    return {"status": "success", "user": current_user}

@app.post("/api/auth/update-profile")
async def update_profile(
    name: str = Form(...), 
    password: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if password and password.strip():
        password_hash, salt = hash_password(password)
        cursor.execute(
            "UPDATE users SET name = ?, password_hash = ?, salt = ? WHERE id = ?",
            (name, password_hash, salt, current_user["id"])
        )
    else:
        cursor.execute(
            "UPDATE users SET name = ? WHERE id = ?",
            (name, current_user["id"])
        )
    conn.commit()
    conn.close()
    return {"status": "success", "message": "Profile updated successfully", "name": name}


# ─────────────────────────────────────────────────────────────
# VERIFICATION API
# ─────────────────────────────────────────────────────────────

@app.post("/api/verify/live-frame")
async def verify_live_frame(
    image: str = Form(...),
    prev_face_json: Optional[str] = Form(None),
    prev_ear: Optional[float] = Form(None),
    track_history: bool = Form(True),
    current_user: dict = Depends(get_current_user)
):
    try:
        # Decode base64 image
        header, encoded = image.split(",", 1) if "," in image else ("", image)
        image_data = base64.b64decode(encoded)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image frame data")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode frame: {str(e)}")

    # Parse prev face bounding box
    prev_face = None
    if prev_face_json:
        try:
            prev_face = json.loads(prev_face_json) # expected list [x, y, w, h]
        except:
            pass

    # Run detectors
    liv_res, face, ear = liveness_check(frame, prev_face, prev_ear)
    eb_score, ea = detect_eyebrow_anomaly(frame, track_history=track_history)
    df_res = analyze_frame(frame)

    # Convert face numpy types to python native types
    face_list = [int(val) for val in face] if face is not None else None
    ear_val = float(ear) if ear is not None else None

    return {
        "face": face_list,
        "ear": ear_val,
        "liveness": bool(liv_res["liveness"]),
        "liveness_reason": liv_res["reason"],
        "eb_score": float(eb_score),
        "eb_analysis": ea,
        "fake_prob": float(df_res["fake_probability"]),
        "confidence": float(df_res["confidence"])
    }

@app.post("/api/verify/image")
async def verify_image(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    try:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file format")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image: {str(e)}")

    # Perform analysis
    report = full_image_analysis(img)
    
    # Save to history
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO history (user_id, type, file_name, verdict, confidence, details_json) VALUES (?, ?, ?, ?, ?, ?)",
        (current_user["id"], "image", file.filename, report["verdict"], report["final_score"], json.dumps(report))
    )
    conn.commit()
    conn.close()

    return report

@app.post("/api/verify/video")
async def verify_video(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    # Save the file temporarily
    temp_filename = f"temp_{uuid.uuid4().hex}_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        reset_frame_history()
        cap = cv2.VideoCapture(temp_filename)
        fps_v = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = max(1, int(fps_v * 3))

        eb_scores, fake_probs, liveness_flags = [], [], []
        prev_face = prev_ear = None
        n_samples = 0

        for i in range(0, total_frames, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: 
                break
            eb_score, _ = detect_eyebrow_anomaly(frame, track_history=True)
            liv_res, face, ear = liveness_check(frame, prev_face, prev_ear)
            df_res = analyze_frame(frame)
            prev_face, prev_ear = face, ear
            
            eb_scores.append(eb_score)
            fake_probs.append(df_res["fake_probability"])
            liveness_flags.append(liv_res["liveness"])
            n_samples += 1
            if n_samples >= 30: 
                break

        cap.release()
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    if not eb_scores:
        raise HTTPException(status_code=400, detail="Could not read or process frames from the video")

    avg_eb = np.mean(eb_scores)
    avg_fp = np.mean(fake_probs)
    blink_pct = sum(liveness_flags) / len(liveness_flags)
    
    # Combined score
    liveness_ok = blink_pct >= 0.15
    fv = avg_eb * 0.45 + (0.9 if liveness_ok else 0.05) * 0.30 + (1 - avg_fp) * 0.25
    
    if fv >= 0.65:
        verdict = "REAL"
    elif fv >= 0.48:
        verdict = "SUSPICIOUS"
    else:
        verdict = "FAKE"
        
    report = {
        "verdict": verdict,
        "confidence": round(float(fv), 3),
        "avg_eyebrow": round(float(avg_eb), 3),
        "liveness_rate": round(float(blink_pct), 3),
        "avg_deepfake": round(float(avg_fp), 3),
        "frames_analyzed": n_samples
    }

    # Save to history
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO history (user_id, type, file_name, verdict, confidence, details_json) VALUES (?, ?, ?, ?, ?, ?)",
        (current_user["id"], "video", file.filename, verdict, fv, json.dumps(report))
    )
    conn.commit()
    conn.close()

    return report

# ─────────────────────────────────────────────────────────────
# HISTORY API
# ─────────────────────────────────────────────────────────────

@app.get("/api/history")
async def get_history(current_user: dict = Depends(get_current_user)):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, type, file_name, verdict, confidence, details_json, created_at FROM history WHERE user_id = ? ORDER BY created_at DESC",
        (current_user["id"],)
    )
    rows = cursor.fetchall()
    conn.close()
    
    history_list = []
    for r in rows:
        item = dict(r)
        try:
            item["details"] = json.loads(item["details_json"])
        except:
            item["details"] = {}
        del item["details_json"]
        history_list.append(item)
        
    return {"status": "success", "history": history_list}

@app.post("/api/history/save")
async def save_history(
    type: str = Form(...),
    file_name: str = Form(...),
    verdict: str = Form(...),
    confidence: float = Form(...),
    details_json: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO history (user_id, type, file_name, verdict, confidence, details_json) VALUES (?, ?, ?, ?, ?, ?)",
        (current_user["id"], type, file_name, verdict, confidence, details_json)
    )
    conn.commit()
    conn.close()
    return {"status": "success", "message": "History saved successfully"}


# ─────────────────────────────────────────────────────────────
# STATIC FILES SERVING & FRONTEND ENTRYPOINT
# ─────────────────────────────────────────────────────────────

# Ensure the static files directory exists
os.makedirs("static", exist_ok=True)

# Mount the static directory
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
