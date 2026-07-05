"""
image_analysis.py  v3.0 — calibrated AI detection
Run: python3 -c "from image_analysis import full_image_analysis; print('OK')"
"""
import cv2, numpy as np, io, importlib, sys

__version__ = "3.0"

try:
    from PIL import Image as _PIL
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

_fc  = cv2.CascadeClassifier(
           cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
_ec  = cv2.CascadeClassifier(
           cv2.data.haarcascades + 'haarcascade_eye.xml')


# ── helpers ──────────────────────────────────────────────────
def _resize(frame, size=512):
    h, w = frame.shape[:2]
    s = size / max(h, w, 1)
    if s < 1.0:
        return cv2.resize(frame,(int(w*s),int(h*s)),interpolation=cv2.INTER_AREA)
    return frame

def _face(gray):
    for scale in [1.05,1.1,1.15,1.2]:
        f = _fc.detectMultiScale(gray,scale,4,minSize=(40,40))
        if len(f): return max(f, key=lambda x:x[2]*x[3])
    return None

def _dot(s):
    return "🟢" if s>=0.70 else ("🟡" if s>=0.50 else "🔴")


# ────────────────────────────────────────────────────────────
# T1  EYEBROW PATTERN
# ────────────────────────────────────────────────────────────
def _t1_eyebrow(frame):
    try:
        eb = importlib.import_module("eyebrow")
        eb.reset_frame_history()
        score, ea = eb.detect_eyebrow_anomaly(frame, track_history=False)
        return {"score": float(score),
                "detail": {
                    "shape":ea["shape"],"density":ea["density"],
                    "continuity":ea["continuity"],"texture":ea["texture"],
                    "symmetry":ea["symmetry"],"lighting":ea["lighting"],
                    "quality":ea["quality"],"angle":ea["angle"],
                    "visibility":ea["visibility"],"full_analysis":ea,
                    "verdict": f"Shape {ea['shape']*100:.0f}% | Density {ea['density']*100:.0f}% | Texture {ea['texture']*100:.0f}%"
                }}
    except Exception as e:
        return {"score":0.50,"detail":{"verdict":f"Eyebrow module error: {e}","full_analysis":{}}}


# ────────────────────────────────────────────────────────────
# T2  FACIAL SYMMETRY
# Real human: 0.76–0.88   AI: often > 0.91
# ────────────────────────────────────────────────────────────
def _t2_symmetry(frame):
    img  = _resize(frame)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = _face(gray)

    if face is None:
        return {"score":0.35,"detail":{"raw_symmetry":0.0,
                "verdict":"No face detected — AI illustration likely"}}

    x,y,w,h = face
    roi  = cv2.resize(gray[y:y+h,x:x+w],(128,128))
    L    = roi[:,:64]
    R    = cv2.flip(roi[:,64:],1)
    raw  = float(1.0 - np.mean(cv2.absdiff(L,R))/255.0)

    if   raw > 0.94: s=0.10; v=f"Perfect symmetry {raw*100:.1f}% — strong AI"
    elif raw > 0.91: s=0.32; v=f"High symmetry {raw*100:.1f}% — likely AI"
    elif raw > 0.88: s=0.55; v=f"Slightly high {raw*100:.1f}% — borderline"
    elif raw >= 0.76:s=0.92; v=f"Natural {raw*100:.1f}% — real human range"
    elif raw >= 0.60:s=0.68; v=f"Low symmetry {raw*100:.1f}% — angled face"
    else:            s=0.40; v=f"Very asymmetric {raw*100:.1f}%"

    return {"score":round(s,2),"detail":{"raw_symmetry":round(raw,3),"verdict":v}}


# ────────────────────────────────────────────────────────────
# T3  SKIN / COLOR AUTHENTICITY
# Proven signals from diagnostic:
#   saturation  36   → AI (real portrait 70-180)
#   channel corr 0.97→ AI (real < 0.92)
#   ch-noise CV 0.032→ AI (real > 0.08)
#   JPEG ghost  2.08 → AI (real > 5)
# ────────────────────────────────────────────────────────────
def _t3_skin(frame):
    img  = _resize(frame)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat  = float(np.mean(hsv[:,:,1]))

    # channel correlation
    B,G,R = [img[:,:,i].flatten().astype(float) for i in range(3)]
    corr  = float((np.corrcoef(R,G)[0,1] + np.corrcoef(R,B)[0,1]) / 2)

    # per-channel noise uniformity
    chn = []
    for i in range(3):
        c = img[:,:,i].astype(np.float32)
        chn.append(float(np.std(c - cv2.GaussianBlur(c,(5,5),0))))
    ch_cv = float(np.std(chn)/(np.mean(chn)+1e-9))

    # JPEG ghost
    ghost = 7.0
    if _PIL_OK:
        try:
            p   = _PIL.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO(); p.save(buf,'JPEG',quality=70); buf.seek(0)
            q   = cv2.imdecode(np.frombuffer(buf.read(),np.uint8),cv2.IMREAD_COLOR)
            ghost = float(np.mean(np.abs(img.astype(float)-q.astype(float))))
        except: pass

    # micro-texture on cheek
    face = _face(gray)
    mt   = 50.0
    if face is not None:
        x,y,w,h = face
        sk = gray[y+int(h*.38):y+int(h*.62), x+int(w*.18):x+int(w*.82)]
        if sk.size > 100:
            bsz=6
            v_=[np.var(sk[r:r+bsz,c:c+bsz].astype(float))
                for r in range(0,sk.shape[0]-bsz,bsz)
                for c in range(0,sk.shape[1]-bsz,bsz)]
            mt = float(np.mean(v_)) if v_ else 50.0

    # ── score each signal ────────────────────────────────────
    # saturation  (diagnostic: 36 = AI)
    if   sat < 45:  ss=0.05; sv=f"Sat={sat:.0f} — extremely low (AI desaturation)"
    elif sat < 65:  ss=0.25; sv=f"Sat={sat:.0f} — below real portrait range"
    elif sat < 80:  ss=0.52; sv=f"Sat={sat:.0f} — borderline"
    elif sat <=175: ss=0.92; sv=f"Sat={sat:.0f} — natural"
    else:           ss=0.20; sv=f"Sat={sat:.0f} — hyper-saturated (AI)"

    # channel correlation  (diagnostic: 0.97 = AI)
    if   corr > 0.96: cs=0.05; cv2_v=f"Corr={corr:.3f} — channels identical (AI)"
    elif corr > 0.93: cs=0.28; cv2_v=f"Corr={corr:.3f} — over-correlated (likely AI)"
    elif corr > 0.90: cs=0.52; cv2_v=f"Corr={corr:.3f} — borderline"
    else:             cs=0.92; cv2_v=f"Corr={corr:.3f} — natural diversity"

    # per-channel noise  (diagnostic: 0.032 = AI)
    if   ch_cv < 0.04: ns2=0.05; nv=f"Ch-noise CV={ch_cv:.3f} — channels identical (AI)"
    elif ch_cv < 0.07: ns2=0.32; nv=f"Ch-noise CV={ch_cv:.3f} — low diversity"
    else:              ns2=0.90; nv=f"Ch-noise CV={ch_cv:.3f} — natural"

    # JPEG ghost  (diagnostic: 2.08 = AI)
    if   ghost < 2.5:  gs=0.05; gv=f"Ghost={ghost:.2f} — no prior compression (AI)"
    elif ghost < 4.0:  gs=0.30; gv=f"Ghost={ghost:.2f} — very low (likely AI)"
    elif ghost < 6.0:  gs=0.60; gv=f"Ghost={ghost:.2f} — borderline"
    else:              gs=0.92; gv=f"Ghost={ghost:.2f} — natural camera history"

    # micro-texture
    if   mt < 20:  ts=0.20
    elif mt < 60:  ts=0.55
    else:          ts=0.88

    # combine  (sat+corr+ghost are most reliable)
    score = ss*0.30 + cs*0.28 + gs*0.25 + ns2*0.12 + ts*0.05
    score = round(float(max(0.0,min(1.0,score))),2)

    verdict = f"{sv} | {cv2_v} | {gv}"
    return {"score":score,
            "detail":{"sat":round(sat,1),"corr":round(corr,3),
                      "ch_cv":round(ch_cv,3),"ghost":round(ghost,2),
                      "micro_texture":round(mt,1),"verdict":verdict}}


# ────────────────────────────────────────────────────────────
# T4  EDGE + LOCAL CONTRAST
# local_contrast_cv: AI 0.5-0.8, real > 0.85
# ────────────────────────────────────────────────────────────
def _t4_edge(frame):
    img  = _resize(frame)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    edges = cv2.Canny(blur,40,120)
    ed    = float(np.sum(edges>0)/edges.size)

    sx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=3)
    sy = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=3)
    mag= np.sqrt(sx**2+sy**2)

    ang  = np.arctan2(sy,sx)
    hist,_= np.histogram(ang[mag>4],bins=36,range=(-np.pi,np.pi))
    hist  = hist/(hist.sum()+1e-9)
    ent   = float(-np.sum(hist*np.log(hist+1e-9))/3.58)

    step=32
    lc = [float(np.std(gray[r:r+step,c:c+step].astype(float)))
          for r in range(0,gray.shape[0]-step,step)
          for c in range(0,gray.shape[1]-step,step)]
    lc_cv = float(np.std(lc)/(np.mean(lc)+1e-9)) if lc else 0.5

    # local contrast score
    if   lc_cv < 0.50: lcs=0.12; lv=f"LC-CV={lc_cv:.3f} — uniform (AI)"
    elif lc_cv < 0.72: lcs=0.42; lv=f"LC-CV={lc_cv:.3f} — low variety (possible AI)"
    elif lc_cv < 0.88: lcs=0.72; lv=f"LC-CV={lc_cv:.3f} — moderate"
    else:              lcs=0.92; lv=f"LC-CV={lc_cv:.3f} — natural"

    # edge score
    if   ed < 0.025:        es=0.18; ev=f"ED={ed*100:.1f}% — too smooth (AI)"
    elif ent < 0.68:        es=0.25; ev=f"Entropy={ent:.2f} — low direction variety"
    elif 0.04<=ed<=0.22:   es=0.88; ev=f"ED={ed*100:.1f}% entropy={ent:.2f} — natural"
    else:                   es=0.52; ev=f"ED={ed*100:.1f}% — borderline"

    score = round(float(max(0.0,min(1.0,es*0.40 + lcs*0.60))),2)
    return {"score":score,
            "detail":{"edge_density":round(ed,4),"entropy":round(ent,3),
                      "lc_cv":round(lc_cv,3),
                      "verdict":f"{ev} | {lv}"}}


# ────────────────────────────────────────────────────────────
# T5  NOISE PATTERN
# ────────────────────────────────────────────────────────────
def _t5_noise(frame):
    img  = _resize(frame)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ns   = float(np.std(gray - cv2.GaussianBlur(gray,(5,5),0)))

    F     = np.fft.fftshift(np.fft.fft2(gray))
    mag   = np.abs(F); h,w = mag.shape
    outer = mag.copy(); outer[h//2-18:h//2+18,w//2-18:w//2+18]=0
    pr    = float(outer.max()/(mag[h//2-4:h//2+4,w//2-4:w//2+4].mean()+1e-9))

    bsz=8
    dv=[float(np.var(cv2.dct(gray[r:r+bsz,c:c+bsz])))
        for r in range(0,gray.shape[0]-bsz,bsz)
        for c in range(0,gray.shape[1]-bsz,bsz)]
    dcv=float(np.std(dv)/(np.mean(dv)+1e-9)) if dv else 0

    if   ns<2.0:     ns2=0.08; nv=f"σ={ns:.1f} — near-zero (AI over-smooth)"
    elif ns<4.0:     ns2=0.32; nv=f"σ={ns:.1f} — below camera threshold"
    elif ns<=20:     ns2=0.85; nv=f"σ={ns:.1f} — natural camera noise"
    elif ns<=28:     ns2=0.62; nv=f"σ={ns:.1f} — moderate"
    else:            ns2=0.42; nv=f"σ={ns:.1f} — high noise"

    if pr>1100:      ns2=min(ns2,0.28)
    if dcv<0.4:      ds=0.28; dv2=f"DCT-CV={dcv:.2f} — uniform blocks (AI)"
    else:            ds=0.82; dv2=f"DCT-CV={dcv:.2f} — natural"

    score=round(float(max(0.0,min(1.0,ns2*0.55+ds*0.45))),2)
    return {"score":score,
            "detail":{"noise_std":round(ns,2),"peak_ratio":round(pr,1),
                      "dct_cv":round(dcv,3),
                      "verdict":f"{nv} | {dv2}"}}


# ────────────────────────────────────────────────────────────
# T6  COLOR DISTRIBUTION
# ────────────────────────────────────────────────────────────
def _t6_color(frame):
    img  = _resize(frame)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hstd = float(np.std(hsv[:,:,0].astype(float)))
    sat  = float(np.mean(hsv[:,:,1].astype(float)))
    sstd = float(np.std(hsv[:,:,1].astype(float)))

    B,G,R=[img[:,:,i].flatten().astype(float) for i in range(3)]
    corr =float((np.corrcoef(R,G)[0,1]+np.corrcoef(R,B)[0,1])/2)

    # hue spread
    if   hstd<12:    hs=0.18; hv=f"Hue-σ={hstd:.0f} — compressed (AI)"
    elif hstd<18:    hs=0.42; hv=f"Hue-σ={hstd:.0f} — narrow"
    elif hstd<=70:   hs=0.88; hv=f"Hue-σ={hstd:.0f} — natural"
    else:            hs=0.55; hv=f"Hue-σ={hstd:.0f} — wide"

    # saturation (most diagnostic)
    if   sat<45:     ss=0.05; sv=f"Sat={sat:.0f} — AI desaturation"
    elif sat<65:     ss=0.28; sv=f"Sat={sat:.0f} — below portrait range"
    elif sat<80:     ss=0.52; sv=f"Sat={sat:.0f} — borderline"
    elif sat<=175:   ss=0.92; sv=f"Sat={sat:.0f} — natural"
    else:            ss=0.18; sv=f"Sat={sat:.0f} — hyper-saturated (AI)"

    # channel correlation
    if   corr>0.96:  cs=0.05; cv_=f"Corr={corr:.3f} — identical (AI)"
    elif corr>0.93:  cs=0.28; cv_=f"Corr={corr:.3f} — over-correlated"
    elif corr>0.90:  cs=0.52; cv_=f"Corr={corr:.3f} — borderline"
    else:            cs=0.90; cv_=f"Corr={corr:.3f} — natural"

    score=round(float(max(0.0,min(1.0,hs*0.20+ss*0.50+cs*0.30))),2)
    return {"score":score,
            "detail":{"hue_std":round(hstd,1),"sat_mean":round(sat,1),
                      "sat_std":round(sstd,1),"avg_corr":round(corr,3),
                      "verdict":f"{sv} | {hv} | {cv_}"}}


# ────────────────────────────────────────────────────────────
# MASTER
# ────────────────────────────────────────────────────────────
def full_image_analysis(frame):
    t = {
        "eyebrow":  _t1_eyebrow(frame),
        "symmetry": _t2_symmetry(frame),
        "skin":     _t3_skin(frame),
        "edge":     _t4_edge(frame),
        "noise":    _t5_noise(frame),
        "color":    _t6_color(frame),
    }

    # skin+ghost carry the most reliable AI signals
    fs = (t["eyebrow"]["score"]  * 0.18 +
          t["skin"]["score"]     * 0.27 +
          t["noise"]["score"]    * 0.20 +
          t["color"]["score"]    * 0.18 +
          t["symmetry"]["score"] * 0.10 +
          t["edge"]["score"]     * 0.07)
    fs = round(float(fs), 3)

    flags = sum(1 for k in t if t[k]["score"] < 0.50)

    if   fs >= 0.68 and flags == 0: verdict="REAL";      reason=f"All 6 techniques confirm authentic content"
    elif fs >= 0.62 and flags <= 1: verdict="REAL";      reason=f"Analysis confirms authentic content ({flags}/6 flag)"
    elif flags >= 2:                verdict="FAKE";       reason=f"{flags}/6 techniques detected AI indicators"
    else:                           verdict="SUSPICIOUS"; reason=f"Mixed signals — {flags}/6 techniques flagged"

    return {"techniques":t,"final_score":fs,
            "verdict":verdict,"reason":reason,"fake_flags":flags}