"""
app.py — AI-Powered Content Authenticity System
Live Camera : step-by-step instructions shown DURING verification
              1 frame capture every 3 seconds (not continuous flood)
Upload Image: 6 independent advanced techniques (no liveness)
Upload Video: 3 independent checks, each own verdict
"""
import streamlit as st
import cv2
import numpy as np
from utils import load_image
from liveness import liveness_check
from eyebrow import detect_eyebrow_anomaly, reset_frame_history
from deepfake import analyze_frame
import importlib, sys
# Force fresh reload every run — prevents Streamlit from using cached old module
for _m in list(sys.modules.keys()):
    if any(_m == x or _m.startswith(x+'.') for x in ['image_analysis','eyebrow','liveness','deepfake','utils']):
        del sys.modules[_m]

import image_analysis as _ia_mod
importlib.reload(_ia_mod)
from image_analysis import full_image_analysis, __version__ as _IA_VERSION
import os, time

st.set_page_config(page_title="Deepfake Verification System", layout="wide")
st.title("🔍 AI-Powered Content Authenticity System")
st.markdown("**Core: Eyebrow Pattern Analysis + 5 Advanced AI-Detection Techniques**")

mode = st.sidebar.radio("Select Mode", [
    "🎥 Live Camera Verification",
    "📸 Upload Image",
    "🎬 Upload Video"
])

# ─────────────────────────────────────────────────────────────
# SHARED UI HELPERS
# ─────────────────────────────────────────────────────────────
def verdict_banner(label, score_pct, reason):
    if label == "REAL":
        st.success(f"## ✅ RESULT: HUMAN-CREATED / REAL\n**Confidence: {score_pct:.0f}%** — {reason}")
    elif label == "FAKE":
        st.error(f"## ❌ RESULT: AI-GENERATED / FAKE\n**Confidence: {score_pct:.0f}%** — {reason}")
    else:
        st.warning(f"## ⚠️ RESULT: UNCERTAIN — MANUAL REVIEW\n**Confidence: {score_pct:.0f}%** — {reason}")

def dot(score):
    return "🟢" if score >= 0.75 else ("🟡" if score >= 0.52 else "🔴")

def tile(col, icon, label, score, msg):
    col.metric(f"{icon} {label}", f"{dot(score)} {int(score*100)}%")
    col.caption(msg)

def avg_tile(col, icon, label, scores):
    avg = np.mean(scores) if scores else 0
    col.metric(f"{icon} {label}", f"{dot(avg)} {avg*100:.0f}%")

def render_eyebrow_panel(ea):
    st.markdown("**🔎 5 Quality Factors**")
    c1,c2,c3,c4,c5 = st.columns(5)
    tile(c1,"💡","Lighting",     ea["lighting"],   ea["lighting_msg"])
    tile(c2,"📷","Image Quality",ea["quality"],    ea["quality_msg"])
    tile(c3,"📐","Face Angle",   ea["angle"],      ea["angle_msg"])
    tile(c4,"👤","Face Visible", ea["visibility"], ea["visibility_msg"])
    tile(c5,"🎞️","Consistency",  ea["consistency"],ea["consistency_msg"])
    st.markdown("**🧬 5 Eyebrow Features**")
    f1,f2,f3,f4,f5 = st.columns(5)
    tile(f1,"🌀","Shape",       ea["shape"],     ea["shape_msg"])
    tile(f2,"🔬","Hair Density",ea["density"],   ea["density_msg"])
    tile(f3,"〰️","Continuity",  ea["continuity"],ea["continuity_msg"])
    tile(f4,"🪡","Texture",     ea["texture"],   ea["texture_msg"])
    tile(f5,"⚖️","Symmetry",    ea["symmetry"],  ea["symmetry_msg"])


# ═════════════════════════════════════════════════════════════
# MODE 1 — LIVE CAMERA
# 1 capture every 3 seconds | step-by-step instructions live
# ═════════════════════════════════════════════════════════════
if mode == "🎥 Live Camera Verification":
    st.write("### 👤 Live Person Verification")

    # Session state init
    for key, default in [
        ("cam_running", False), ("cam_results", []),
        ("cam_analyses", []),   ("cam_step", 0),
        ("blink_count", 0),     ("turn_done", False),
        ("last_capture", 0.0),  ("capture_count", 0),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    col_btn1, col_btn2 = st.columns(2)
    start = col_btn1.button("▶️ Start Verification", type="primary")
    stop  = col_btn2.button("⏹️ Stop & Show Results", type="secondary")

    if start:
        st.session_state.cam_running   = True
        st.session_state.cam_results   = []
        st.session_state.cam_analyses  = []
        st.session_state.cam_step      = 0
        st.session_state.blink_count   = 0
        st.session_state.turn_done     = False
        st.session_state.last_capture  = 0.0
        st.session_state.capture_count = 0
        reset_frame_history()

    if stop:
        st.session_state.cam_running = False

    # Layout: camera left | step instructions right
    cam_col, inst_col = st.columns([3, 2])
    with cam_col:
        frame_win = st.empty()
    with inst_col:
        step_ph   = st.empty()
        status_ph = st.empty()
        tip_ph    = st.empty()

    # ── Step Definitions ────────────────────────────────────
    STEPS = [
        {
            "num": "Step 1 of 4", "icon": "👤",
            "title": "Position Your Face",
            "color": "info",
            "instructions": [
                "📐 Look directly at the camera",
                "👤 Keep BOTH eyebrows fully visible",
                "💡 Face a lamp or window for good lighting",
                "📏 Sit 30–60 cm from the camera",
                "🚫 Do not wear a hat or cover your forehead",
            ],
            "waiting": "⏳ Waiting to detect your face…",
            "done": "✅ Face detected! Proceeding to Step 2…"
        },
        {
            "num": "Step 2 of 4", "icon": "👁️",
            "title": "Blink Naturally",
            "color": "warning",
            "instructions": [
                "👁️ Blink both eyes naturally",
                "🔁 Blink slowly 2–3 times",
                "⏸️ Hold eyes OPEN between blinks",
                "❌ Do NOT squint or wink — blink normally",
                "⏱️ Take your time, no rush",
            ],
            "waiting": "⏳ Waiting to detect your blink…",
            "done": "✅ Blink confirmed! Proceeding to Step 3…"
        },
        {
            "num": "Step 3 of 4", "icon": "↔️",
            "title": "Slowly Turn Your Head",
            "color": "warning",
            "instructions": [
                "⬅️ Slowly turn head LEFT (15–30°)",
                "➡️ Then slowly turn head RIGHT (15–30°)",
                "🐢 Move slowly and smoothly",
                "👤 Keep face in frame throughout",
                "📐 Return to centre when done",
            ],
            "waiting": "⏳ Waiting to detect head movement…",
            "done": "✅ Head movement confirmed! Proceeding to Step 4…"
        },
        {
            "num": "Step 4 of 4", "icon": "🌀",
            "title": "Hold Still — Eyebrow Scan",
            "color": "success",
            "instructions": [
                "🌀 HOLD YOUR FACE COMPLETELY STILL",
                "📐 Face directly at the camera",
                "💡 Check that lighting is even on your face",
                "👁️ Keep eyes open, look straight ahead",
                "⏳ Stay still for 5–6 seconds for the scan",
            ],
            "waiting": "🔍 Scanning eyebrow patterns… hold still",
            "done": "✅ Eyebrow scan complete!"
        },
    ]

    def render_step(idx, phase="waiting"):
        s = STEPS[idx]
        msg = s["done"] if phase == "done" else (
              s["waiting"] if phase == "waiting_only" else
              "\n".join(f"• {i}" for i in s["instructions"]))
        header = f"**{s['icon']} {s['num']} — {s['title']}**"
        content = f"{header}\n\n{msg}"
        if phase == "done":
            step_ph.success(content)
        elif s["color"] == "info":
            step_ph.info(content)
        elif s["color"] == "warning":
            step_ph.warning(content)
        else:
            step_ph.success(content)

    # ── LIVE CAPTURE LOOP (60 fps display, analyse every 1 sec) ──
    if st.session_state.cam_running:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

        prev_face = prev_ear = None
        frame_count = 0
        CAPTURE_INTERVAL = 1.0   # analyse once per second; display every frame

        render_step(0)

        while st.session_state.cam_running:
            ret, frame = cap.read()
            if not ret:
                tip_ph.error("❌ Camera not accessible.")
                break

            # ── Show EVERY frame for smooth 60 fps video ────
            frame_win.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                            use_column_width=True)

            now  = time.time()
            step = st.session_state.cam_step

            # ── Analyse once per second (not every frame) ───
            if now - st.session_state.last_capture >= CAPTURE_INTERVAL:
                st.session_state.last_capture  = now
                st.session_state.capture_count += 1

                # Run all detectors on this captured frame
                liv_res, face, ear = liveness_check(frame, prev_face, prev_ear)
                prev_face, prev_ear = face, ear
                liveness_ok = liv_res["liveness"]

                eb_score, ea = detect_eyebrow_anomaly(frame,
                                                       track_history=(step == 3))
                df_res    = analyze_frame(frame)
                fake_prob = df_res["fake_probability"]

                st.session_state.cam_results.append({
                    "liveness": liveness_ok,
                    "eb_score": eb_score,
                    "fake_prob": fake_prob,
                })
                st.session_state.cam_analyses.append(ea)

                # ── Step progression logic ───────────────────
                if step == 0:
                    render_step(0)
                    tip_ph.info("👤 Position your face in the frame")
                    if face is not None:
                        render_step(0, "done")
                        time.sleep(1.0)
                        st.session_state.cam_step = 1

                elif step == 1:
                    render_step(1)
                    if liveness_ok:
                        st.session_state.blink_count += 1
                    if st.session_state.blink_count >= 2:
                        render_step(1, "done")
                        time.sleep(1.0)
                        st.session_state.cam_step = 2
                    else:
                        tip_ph.info(
                            f"👁️ Blink detected: {st.session_state.blink_count}/2 — "
                            f"keep blinking naturally")

                elif step == 2:
                    render_step(2)
                    if liveness_ok:
                        st.session_state.turn_done = True
                    if st.session_state.turn_done:
                        render_step(2, "done")
                        time.sleep(1.0)
                        st.session_state.cam_step = 3
                    else:
                        tip_ph.info("↔️ Slowly turn your head left then right")

                elif step == 3:
                    render_step(3)
                    cap_so_far = len(st.session_state.cam_results)
                    tip_ph.success(
                        f"🌀 Scanning eyebrow patterns… "
                        f"({cap_so_far} capture{'s' if cap_so_far!=1 else ''} done, "
                        f"need 3 more still captures)")
                    if cap_so_far >= 5:      # need at least 5 captures total
                        render_step(3, "done")
                        time.sleep(0.5)
                        st.session_state.cam_running = False
                        break

                # ── Live readings panel ──────────────────────
                n = len(st.session_state.cam_results)
                lines = [
                    f"👤 Face:        {'✅ Yes' if face else '❌ Not found'}",
                    f"👁️ Blinks:      {st.session_state.blink_count}/2",
                    f"↔️  Head turn:   {'✅ Done' if st.session_state.turn_done else '⏳ Pending'}",
                    f"🌀 Eyebrows:    {'✅' if eb_score>=0.6 else '⚠️'} {eb_score*100:.0f}%",
                    f"💡 Lighting:    {'✅' if ea['lighting']>=0.6 else '⚠️'} {ea['lighting']*100:.0f}%",
                    f"📐 Angle:       {'✅' if ea['angle']>=0.65 else '⚠️'} {ea['angle']*100:.0f}%",
                    f"🤖 Deepfake:    {'⚠️ High' if fake_prob>0.6 else '✅ Low'} ({fake_prob*100:.0f}%)",
                    f"🎞️ Captures:    {n}  (1 per sec)",
                ]
                status_ph.code("\n".join(lines))

                # Context tips (only override tip_ph if not in specific step)
                if step == 0 and face is None:
                    tip_ph.warning("👤 Move your face into the camera frame")
                elif ea["lighting"] < 0.52:
                    if step not in (1, 2):
                        tip_ph.warning("💡 Move to better lighting — your face is too dark")
                elif ea["angle"] < 0.52 and step == 3:
                    tip_ph.warning("📐 Face the camera more directly for better eyebrow scan")

            frame_count += 1

        cap.release()

    # ── RESULTS after stop ───────────────────────────────────
    results  = st.session_state.cam_results
    analyses = st.session_state.cam_analyses

    if results and not st.session_state.cam_running:
        st.write("---")
        st.subheader("📊 Verification Results")
        st.caption(f"Based on {len(results)} captures taken 1 per 3 seconds")

        n           = len(results)
        avg_eb      = np.mean([r["eb_score"]  for r in results])
        avg_fp      = np.mean([r["fake_prob"] for r in results])
        blinks      = sum(1 for r in results if r["liveness"])
        liveness_ok = blinks >= 2 or st.session_state.turn_done

        # ── CHECK 1: LIVENESS ─────────────────────────────
        st.markdown("### 🔵 Check 1 — Liveness Detection")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Blinks Detected",  f"{blinks}")
            st.metric("Head Turn",        "✅ Yes" if st.session_state.turn_done else "❌ No")
            st.metric("Captures Taken",   n)
        with c2:
            if liveness_ok:
                verdict_banner("REAL", min(blinks/max(n,1)*100*3, 97),
                               "Blink and head movement confirmed — live person verified")
            else:
                verdict_banner("FAKE", 88,
                               "No blink or head movement detected — possible photo/replay attack")

        # ── CHECK 2: EYEBROW ──────────────────────────────
        st.markdown("---")
        st.markdown("### 🟣 Check 2 — Eyebrow Pattern Analysis (Main Feature)")
        if analyses:
            render_eyebrow_panel(analyses[-1])
        st.markdown("")
        if avg_eb >= 0.70:
            verdict_banner("REAL", avg_eb*100,
                           "All 5 eyebrow features match natural human patterns")
        elif avg_eb >= 0.55:
            verdict_banner("SUSPICIOUS", avg_eb*100,
                           "Some eyebrow anomalies — possible deepfake or partial edit")
        else:
            verdict_banner("FAKE", avg_eb*100,
                           "Eyebrow patterns strongly inconsistent with real human face")

        # ── CHECK 3: DEEPFAKE ─────────────────────────────
        st.markdown("---")
        st.markdown("### 🟠 Check 3 — Deepfake Detection")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Avg Fake Probability", f"{avg_fp*100:.0f}%")
            st.metric("Authenticity Score",   f"{(1-avg_fp)*100:.0f}%")
            st.metric("Captures Analysed",    n)
        with c2:
            if avg_fp < 0.35:
                verdict_banner("REAL", (1-avg_fp)*100, "No deepfake indicators detected")
            elif avg_fp < 0.60:
                verdict_banner("SUSPICIOUS", (1-avg_fp)*100, "Some manipulation indicators present")
            else:
                verdict_banner("FAKE", avg_fp*100, "Strong deepfake indicators detected across captures")

        # ── FINAL COMBINED ────────────────────────────────
        st.markdown("---")
        st.markdown("### 🏁 Final Combined Verdict  *(Eyebrow 45% + Liveness 30% + Deepfake 25%)*")
        fs = avg_eb*0.45 + (0.9 if liveness_ok else 0.05)*0.30 + (1-avg_fp)*0.25
        if fs >= 0.68 and liveness_ok:
            verdict_banner("REAL", fs*100,
                           "All 3 checks confirm a real human person")
        elif fs < 0.50 or not liveness_ok:
            verdict_banner("FAKE", (1-fs)*100,
                           "Multiple checks failed — AI/deepfake or replay attack likely")
        else:
            verdict_banner("SUSPICIOUS", fs*100,
                           "Mixed results — manual review recommended")


# ═════════════════════════════════════════════════════════════
# MODE 2 — UPLOAD IMAGE
# 6 independent advanced techniques, NO liveness
# ═════════════════════════════════════════════════════════════
elif mode == "📸 Upload Image":
    st.write("### 📸 Image Authenticity Analysis")

    uploaded = st.file_uploader("📁 Upload an image", type=["jpg","jpeg","png"])

    if uploaded is not None:
        img = load_image(uploaded)
        col_img, col_res = st.columns([1, 1])

        with col_img:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                     caption="Uploaded Image", use_column_width=True)

        with st.spinner("🔍 Analysing image..."):
            report = full_image_analysis(img)
            time.sleep(0.4)

        core  = report.get("core", {})
        bp    = core.get("bulletproof_count", 0)
        ghost = core.get("ghost", 9.0)
        sat   = core.get("sat",  100.0)
        corr  = core.get("corr",  0.80)
        fs    = report["final_score"]
        flags = report["fake_flags"]
        t     = report["techniques"]

        with col_res:
            # ── SINGLE FINAL VERDICT ──────────────────────
            if report["verdict"] == "FAKE":
                st.error("## ❌ AI-GENERATED / FAKE")
            elif report["verdict"] == "REAL":
                st.success("## ✅ HUMAN-CREATED / REAL")
            else:
                st.warning("## ⚠️ UNCERTAIN — Review Needed")

            st.metric("Confidence", f"{fs*100:.0f}%")
            st.write("---")

            # ── KEY EVIDENCE (compact) ─────────────────────
            st.markdown("**Key Evidence:**")
            st.write(f"{'🔴' if ghost < 2.5 else '🟢'} JPEG Ghost = **{ghost:.2f}** "
                     f"{'(AI — no prior compression)' if ghost<2.5 else '(Natural)'}")
            st.write(f"{'🔴' if sat < 50 else '🟢'} Saturation = **{sat:.0f}** "
                     f"{'(AI — real portraits: 70–180)' if sat<50 else '(Natural)'}")
            st.write(f"{'🔴' if corr > 0.95 else '🟢'} Channel Corr = **{corr:.3f}** "
                     f"{'(AI — channels identical)' if corr>0.95 else '(Natural)'}")
            st.write(f"{'🔴' if t['eyebrow']['score']<0.55 else '🟢'} Eyebrow Pattern = "
                     f"**{t['eyebrow']['score']*100:.0f}%** "
                     f"{'(Anomaly detected)' if t['eyebrow']['score']<0.55 else '(Natural)'}")

            st.write("---")
            st.caption(f"Techniques flagged: {flags}/6   |   AI signals: {bp}/3")

elif mode == "🎬 Upload Video":
    st.write("### 🎬 Video Authenticity Analysis")

    uploaded = st.file_uploader("📁 Upload a video", type=["mp4","avi","mov"])

    if uploaded is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded.getbuffer())
        st.video("temp_video.mp4")

        with st.spinner("🔍 Analysing video frames..."):
            reset_frame_history()
            cap          = cv2.VideoCapture("temp_video.mp4")
            fps_v        = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_rate  = max(1, int(fps_v * 3))

            eb_scores, fake_probs, liveness_flags = [], [], []
            prev_face = prev_ear = None
            prog = st.progress(0)
            n_samples = 0

            for i in range(0, total_frames, sample_rate):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret: break
                eb_score, _ = detect_eyebrow_anomaly(frame, track_history=True)
                liv_res, face, ear = liveness_check(frame, prev_face, prev_ear)
                df_res = analyze_frame(frame)
                prev_face, prev_ear = face, ear
                eb_scores.append(eb_score)
                fake_probs.append(df_res["fake_probability"])
                liveness_flags.append(liv_res["liveness"])
                n_samples += 1
                prog.progress(min(n_samples / 30, 1.0))
                if n_samples >= 30: break

            cap.release()
            time.sleep(0.3)

        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")

        if eb_scores:
            avg_eb    = np.mean(eb_scores)
            avg_fp    = np.mean(fake_probs)
            blink_pct = sum(liveness_flags) / len(liveness_flags)
            fv        = avg_eb*0.45 + blink_pct*0.30 + (1-avg_fp)*0.25

            # ── SINGLE FINAL VERDICT ──────────────────────
            st.write("---")
            if fv >= 0.65:
                st.success("## ✅ HUMAN-CREATED / REAL")
                verdict_label = "REAL"
            elif fv >= 0.48:
                st.warning("## ⚠️ UNCERTAIN — Manual Review")
                verdict_label = "SUSPICIOUS"
            else:
                st.error("## ❌ AI-GENERATED / FAKE")
                verdict_label = "FAKE"

            st.metric("Confidence", f"{fv*100:.0f}%")
            st.write("---")

            # ── KEY EVIDENCE (compact) ─────────────────────
            st.markdown("**Key Evidence:**")
            st.write(f"{'🟢' if avg_eb>=0.65 else '🔴'} Eyebrow Pattern = **{avg_eb*100:.0f}%** "
                     f"{'(Natural)' if avg_eb>=0.65 else '(Anomaly detected)'}")
            st.write(f"{'🟢' if blink_pct>=0.15 else '🔴'} Liveness Rate = **{blink_pct*100:.0f}%** "
                     f"{'(Movement detected)' if blink_pct>=0.15 else '(No natural movement)'}")
            st.write(f"{'🟢' if avg_fp<0.40 else '🔴'} Deepfake Score = **{avg_fp*100:.0f}%** "
                     f"{'(Clean)' if avg_fp<0.40 else '(Manipulation indicators)'}")

            st.write("---")
            st.caption(f"Frames analysed: {len(eb_scores)}  |  Sample rate: every 3 sec")
        else:
            st.error("❌ Could not read any frames from the video.")