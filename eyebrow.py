import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# ─────────────────────────────────────────────────────────────
# QUALITY FACTOR 1 — LIGHTING
# ─────────────────────────────────────────────────────────────
def assess_lighting(gray):
    """
    Returns (lighting_score 0-1, enhanced_gray, lighting_msg)
    Bad lighting → CLAHE + gamma correction before any analysis.
    """
    mean_brightness = np.mean(gray)
    std_brightness  = np.std(gray)

    # Gamma correction for very dark frames
    if mean_brightness < 60:
        gamma   = 1.8
        table   = np.array([(i / 255.0) ** (1.0 / gamma) * 255
                            for i in range(256)]).astype(np.uint8)
        gray = cv2.LUT(gray, table)
        msg = f"Poor lighting corrected (brightness was {mean_brightness:.0f})"
        score = 0.55
    elif mean_brightness > 200:
        # Over-exposed: darken slightly
        gamma = 0.7
        table = np.array([(i / 255.0) ** (1.0 / gamma) * 255
                          for i in range(256)]).astype(np.uint8)
        gray  = cv2.LUT(gray, table)
        msg   = f"Over-exposure corrected (brightness was {mean_brightness:.0f})"
        score = 0.65
    else:
        msg   = f"Good lighting ({mean_brightness:.0f} brightness, {std_brightness:.0f} contrast)"
        score = min(1.0, 0.7 + std_brightness / 150.0)

    # CLAHE always for local contrast
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)

    return round(score, 2), bilateral, msg


# ─────────────────────────────────────────────────────────────
# QUALITY FACTOR 2 — IMAGE / VIDEO QUALITY
# ─────────────────────────────────────────────────────────────
def assess_image_quality(gray):
    """
    Returns (quality_score 0-1, msg)
    Laplacian variance = sharpness proxy.
    """
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if lap_var < 50:
        score = 0.4
        msg   = f"Low image quality / blurry ({lap_var:.0f} sharpness)"
    elif lap_var < 150:
        score = 0.7
        msg   = f"Acceptable image quality ({lap_var:.0f} sharpness)"
    else:
        score = 1.0
        msg   = f"High image quality ({lap_var:.0f} sharpness)"

    return round(score, 2), msg


# ─────────────────────────────────────────────────────────────
# QUALITY FACTOR 3 — FACE ANGLE / DIRECT ANGLE
# ─────────────────────────────────────────────────────────────
def detect_face_and_angle(frame, enhanced_gray):
    """
    Returns (face_bbox, angle_score 0-1, angle_msg, roi_enhanced)
    Multi-scale cascade for different angles.
    Estimates yaw from eye positions.
    """
    all_faces = []
    for scale in [1.1, 1.2, 1.3, 1.4]:
        faces = face_cascade.detectMultiScale(
            enhanced_gray, scale, 5, minSize=(50, 50))
        if len(faces):
            all_faces.extend(faces)

    if not all_faces:
        return None, 0.0, "No face detected", None

    # Largest face = subject
    face = max(all_faces, key=lambda f: f[2] * f[3])
    (x, y, w, h) = face
    roi = enhanced_gray[y:y+h, x:x+w]

    # Estimate angle via eye symmetry
    eyes = _detect_eyes_multi(roi)
    if len(eyes) >= 2:
        eyes_s = sorted(eyes, key=lambda e: e[0])
        le, re = eyes_s[0], eyes_s[-1]
        dx = re[0] - le[0]
        dy = abs(re[1] - le[1])
        # Yaw approximation: if eyes roughly level → frontal
        angle_deg = np.degrees(np.arctan2(dy, dx)) if dx > 0 else 90
        if angle_deg < 8:
            angle_score = 1.0
            angle_msg   = "Direct front-facing angle ✅"
        elif angle_deg < 18:
            angle_score = 0.80
            angle_msg   = f"Slight angle ({angle_deg:.0f}°) — good for analysis"
        elif angle_deg < 35:
            angle_score = 0.60
            angle_msg   = f"Moderate angle ({angle_deg:.0f}°) — partial analysis"
        else:
            angle_score = 0.40
            angle_msg   = f"Large angle ({angle_deg:.0f}°) — limited eyebrow view"
    else:
        angle_score = 0.50
        angle_msg   = "Could not measure angle (only 1 eye detected)"

    return face, round(angle_score, 2), angle_msg, roi


def _detect_eyes_multi(roi_gray):
    """Multi-scale eye detection for partial face / angle tolerance."""
    all_eyes = []
    for scale in [1.05, 1.1, 1.15, 1.2]:
        eyes = eye_cascade.detectMultiScale(
            roi_gray, scale, 5, minSize=(15, 15))
        all_eyes.extend(eyes)

    # De-duplicate
    unique = []
    for e in all_eyes:
        if not any(abs(e[0]-u[0]) < 10 and abs(e[1]-u[1]) < 10
                   for u in unique):
            unique.append(e)
    return sorted(unique, key=lambda e: e[2]*e[3], reverse=True)[:2]


# ─────────────────────────────────────────────────────────────
# QUALITY FACTOR 4 — FULL FACE VISIBLE / OCCLUSION CHECK
# ─────────────────────────────────────────────────────────────
def check_face_visibility(face, frame_shape, eyes):
    """
    Returns (visibility_score 0-1, visible_regions dict, msg)
    Checks face size vs frame, eye count, and upper-face exposure.
    """
    if face is None:
        return 0.0, {}, "No face"

    (x, y, w, h) = face
    fh, fw = frame_shape[:2]

    # Face must occupy reasonable portion of frame
    face_area_ratio = (w * h) / (fw * fh)

    # Eyes visible?
    eyes_visible = len(eyes)

    # Upper face (eyebrow region) not cropped by frame edge
    eyebrow_zone_top  = y - int(h * 0.2)
    brow_visible = eyebrow_zone_top >= 0

    visible_regions = {
        "eyes_detected": eyes_visible,
        "eyebrow_zone_clear": brow_visible,
        "face_area_ratio": round(face_area_ratio, 3)
    }

    score = 0.0
    msgs  = []

    if face_area_ratio < 0.04:
        score += 0.2; msgs.append("Face too small / far away")
    elif face_area_ratio < 0.10:
        score += 0.6; msgs.append("Face partially visible")
    else:
        score += 0.9; msgs.append("Face clearly visible ✅")

    if eyes_visible >= 2:
        score = min(1.0, score + 0.1)
        msgs.append("Both eyes detected ✅")
    elif eyes_visible == 1:
        msgs.append("Only 1 eye detected ⚠️")
    else:
        score = max(0, score - 0.2)
        msgs.append("Eyes not detected ❌")

    if not brow_visible:
        score = max(0, score - 0.15)
        msgs.append("Eyebrow zone partially cropped ⚠️")
    else:
        msgs.append("Eyebrow zone fully visible ✅")

    return round(min(score, 1.0), 2), visible_regions, " | ".join(msgs)


# ─────────────────────────────────────────────────────────────
# EYEBROW REGION EXTRACTION
# ─────────────────────────────────────────────────────────────
def extract_eyebrow_region(roi_gray, eye):
    (ex, ey, ew, eh) = eye
    margin_top    = int(eh * 0.9)
    margin_bottom = int(eh * 0.15)
    y0 = max(0, ey - margin_top)
    y1 = ey + margin_bottom
    x0 = max(0, ex - int(ew * 0.4))
    x1 = min(roi_gray.shape[1], ex + int(ew * 1.4))
    if y0 >= y1 or x0 >= x1:
        return None
    return roi_gray[y0:y1, x0:x1]


# ─────────────────────────────────────────────────────────────
# EYEBROW FEATURE ANALYSIS
# ─────────────────────────────────────────────────────────────
def analyze_shape(region):
    if region is None or region.size == 0:
        return 0.3, "No region"
    try:
        _, binary = cv2.threshold(region, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.4, "No contour"
        cnt = max(contours, key=cv2.contourArea)
        _, _, cw, ch = cv2.boundingRect(cnt)
        ratio = cw / ch if ch else 0
        if 3 <= ratio <= 7:
            return 0.9, "Natural arch shape"
        elif 2 <= ratio <= 9:
            return 0.7, "Acceptable shape"
        else:
            return 0.4, "Unusual shape"
    except:
        return 0.5, "Shape incomplete"


def analyze_density(region):
    if region is None or region.size == 0:
        return 0.3, "No region"
    try:
        _, binary = cv2.threshold(region, 150, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        density = np.sum(morph > 0) / morph.size
        if 0.12 < density < 0.55:
            return 0.9, "Natural hair density"
        elif 0.05 < density < 0.75:
            return 0.7, "Acceptable density"
        else:
            return 0.3, "Abnormal density"
    except:
        return 0.5, "Density incomplete"


def analyze_continuity(region):
    if region is None or region.size == 0:
        return 0.3, "No region"
    try:
        _, binary = cv2.threshold(region, 150, 255, cv2.THRESH_BINARY_INV)
        good, total = 0, 0
        for row in binary:
            transitions = np.sum(np.diff(row.astype(int)) != 0) // 2
            if 1 <= transitions <= 5:
                good += 1
            total += 1
        ratio = good / total if total else 0
        if ratio > 0.7:
            return 0.9, "Natural continuity"
        elif ratio > 0.5:
            return 0.7, "Good continuity"
        else:
            return 0.4, "Broken continuity"
    except:
        return 0.5, "Continuity incomplete"


def analyze_texture(region):
    """Texture = Laplacian edge variance — too low → artificial, too high → noise."""
    if region is None or region.size == 0:
        return 0.3, "No region"
    try:
        lap_var = cv2.Laplacian(region, cv2.CV_64F).var()
        if 8 < lap_var < 120:
            return 0.9, "Natural texture"
        elif 4 < lap_var < 200:
            return 0.7, "Acceptable texture"
        else:
            return 0.35, "Unnatural texture"
    except:
        return 0.5, "Texture incomplete"


def analyze_symmetry(left, right):
    if left is None or right is None:
        return 0.5, "Insufficient data"
    try:
        h = min(left.shape[0], right.shape[0])
        w = min(left.shape[1], right.shape[1])
        if h <= 0 or w <= 0:
            return 0.5, "Cannot normalize"
        l = cv2.equalizeHist(cv2.resize(left[:h, :w], (60, 40)))
        r = cv2.equalizeHist(cv2.resize(right[:h, :w], (60, 40)))
        diff  = cv2.absdiff(l, r)
        score = 1.0 - np.mean(diff) / 255.0
        return round(max(0, min(1, score)), 2), "Symmetry measured"
    except:
        return 0.5, "Symmetry failed"


# ─────────────────────────────────────────────────────────────
# QUALITY FACTOR 5 — MULTI-FRAME CONSISTENCY
# ─────────────────────────────────────────────────────────────
_frame_history = []   # rolling buffer of per-frame eyebrow scores
MAX_HISTORY    = 30

def update_frame_history(score):
    _frame_history.append(score)
    if len(_frame_history) > MAX_HISTORY:
        _frame_history.pop(0)

def multi_frame_consistency():
    """
    Returns (consistency_score 0-1, msg)
    Low std-dev → stable / consistent appearance → more authentic.
    Only meaningful after ≥5 frames.
    """
    if len(_frame_history) < 5:
        return 0.7, f"Accumulating frames ({len(_frame_history)}/{MAX_HISTORY})"
    std  = np.std(_frame_history)
    mean = np.mean(_frame_history)
    if std < 0.05:
        return 1.0, f"Very consistent across {len(_frame_history)} frames ✅"
    elif std < 0.12:
        return 0.8, f"Reasonably consistent ({len(_frame_history)} frames)"
    else:
        return 0.45, f"Inconsistent patterns ({std:.2f} variance) ⚠️"


def reset_frame_history():
    """Call this when switching input source (new video / new camera session)."""
    _frame_history.clear()


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────
def detect_eyebrow_anomaly(frame, track_history=True):
    """
    Full 5-factor eyebrow analysis.

    Returns:
        overall_score (float 0-1)   — 1.0 = fully authentic
        analysis      (dict)        — all sub-scores and messages
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Factor 1: Lighting ──────────────────────────────────
    light_score, enhanced, light_msg = assess_lighting(gray)

    # ── Factor 2: Image Quality ─────────────────────────────
    quality_score, quality_msg = assess_image_quality(gray)

    # ── Factor 3: Face Angle ────────────────────────────────
    face, angle_score, angle_msg, roi = detect_face_and_angle(frame, enhanced)

    if face is None:
        result = {
            "status":       "No face detected",
            "lighting":     light_score,  "lighting_msg":  light_msg,
            "quality":      quality_score,"quality_msg":   quality_msg,
            "angle":        0.0,          "angle_msg":     "No face",
            "visibility":   0.0,          "visibility_msg":"No face",
            "consistency":  0.7,          "consistency_msg":"N/A",
            "shape":        0.3,          "density":       0.3,
            "continuity":   0.3,          "texture":       0.3,
            "symmetry":     0.3,
            "overall_score":0.35,
            "is_anomaly":   True
        }
        return 0.35, result

    # ── Factor 4: Full Face Visible ─────────────────────────
    eyes = _detect_eyes_multi(roi)
    visibility_score, visible_regions, visibility_msg = check_face_visibility(
        face, frame.shape, eyes)

    # ── Eyebrow Feature Analysis ────────────────────────────
    if len(eyes) >= 2:
        eyes_s = sorted(eyes, key=lambda e: e[0])
        le_region = extract_eyebrow_region(roi, eyes_s[0])
        re_region = extract_eyebrow_region(roi, eyes_s[-1])
    elif len(eyes) == 1:
        le_region = extract_eyebrow_region(roi, eyes[0])
        re_region = None
    else:
        le_region = re_region = None

    shape_score,     shape_msg     = analyze_shape(le_region)
    density_score,   density_msg   = analyze_density(le_region)
    continuity_score,continuity_msg= analyze_continuity(le_region)
    texture_score,   texture_msg   = analyze_texture(le_region)
    symmetry_score,  symmetry_msg  = analyze_symmetry(le_region, re_region)

    # Average right eyebrow into feature scores if available
    if re_region is not None:
        shape_score     = (shape_score     + analyze_shape(re_region)[0])     / 2
        density_score   = (density_score   + analyze_density(re_region)[0])   / 2
        continuity_score= (continuity_score+ analyze_continuity(re_region)[0])/ 2
        texture_score   = (texture_score   + analyze_texture(re_region)[0])   / 2

    # ── Factor 5: Multi-Frame Consistency ───────────────────
    # Compute per-frame eyebrow feature score first
    eyebrow_feature_score = (
        shape_score      * 0.25 +
        density_score    * 0.25 +
        continuity_score * 0.20 +
        texture_score    * 0.15 +
        symmetry_score   * 0.15
    )

    if track_history:
        update_frame_history(eyebrow_feature_score)
    consistency_score, consistency_msg = multi_frame_consistency()

    # ── FINAL WEIGHTED SCORE ────────────────────────────────
    # Quality factors penalise / boost eyebrow feature scores
    quality_factor = (
        light_score   * 0.20 +   # Factor 1: Lighting
        quality_score * 0.20 +   # Factor 2: Image quality
        angle_score   * 0.25 +   # Factor 3: Direct angle
        visibility_score * 0.20 + # Factor 4: Full face visible
        consistency_score * 0.15  # Factor 5: Multi-frame
    )

    # Final = blend of eyebrow features (60%) and quality factors (40%)
    overall_score = eyebrow_feature_score * 0.60 + quality_factor * 0.40
    overall_score = round(max(0.0, min(1.0, overall_score)), 2)

    is_anomaly = overall_score < 0.55

    analysis = {
        "status": "Comprehensive 5-factor eyebrow analysis complete",

        # ── Quality Factors ──
        "lighting":         round(light_score,        2),
        "lighting_msg":     light_msg,
        "quality":          round(quality_score,       2),
        "quality_msg":      quality_msg,
        "angle":            round(angle_score,         2),
        "angle_msg":        angle_msg,
        "visibility":       round(visibility_score,    2),
        "visibility_msg":   visibility_msg,
        "consistency":      round(consistency_score,   2),
        "consistency_msg":  consistency_msg,

        # ── Eyebrow Features ──
        "shape":            round(shape_score,         2),
        "shape_msg":        shape_msg,
        "density":          round(density_score,       2),
        "density_msg":      density_msg,
        "continuity":       round(continuity_score,    2),
        "continuity_msg":   continuity_msg,
        "texture":          round(texture_score,       2),
        "texture_msg":      texture_msg,
        "symmetry":         round(symmetry_score,      2),
        "symmetry_msg":     symmetry_msg,

        "overall_score":    overall_score,
        "is_anomaly":       is_anomaly
    }

    return overall_score, analysis