import os, io, json, joblib, numpy as np, pandas as pd, streamlit as st
from src.utils.text_cleaner import TextCleaner
from PIL import Image
import cv2
import tensorflow as tf

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.utils.image import detect_and_crop_face
from src.recommender.emotion_mapping import EMOTION_ID2NAME, EMOTION_QUERY

st.set_page_config(page_title="MoodMate", page_icon="🎵")

# --- Load CNN model (if present) ---
MODEL_PATH = os.path.join("models", "fer_cnn.keras")
CLASS_JSON = os.path.join("models", "class_names.json")

cnn_model = None
class_names = None
if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_JSON):
    cnn_model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_JSON) as f:
        class_names = json.load(f)

# --- Load recommender artifacts ---
VECT_PATH = os.path.join("models", "tfidf_vectorizer.joblib")
INDEX_PATH = os.path.join("models", "song_index.joblib")
SONGS_PARQUET = os.path.join("models", "songs_clean.parquet")

vec = joblib.load(VECT_PATH) if os.path.exists(VECT_PATH) else None
X = joblib.load(INDEX_PATH) if os.path.exists(INDEX_PATH) else None
songs_df = pd.read_parquet(SONGS_PARQUET) if os.path.exists(SONGS_PARQUET) else None

def recommend_for_emotion(emotion_name, top_k=10):
    if vec is None or X is None or songs_df is None:
        return pd.DataFrame(columns=["title","artist","genre","mood","tags","search_query"])
    query = EMOTION_QUERY.get(emotion_name, "chill balanced")
    qvec = vec.transform([query])
    sims = (qvec @ X.T).toarray().ravel()
    idx = np.argsort(-sims)[:top_k]
    res = songs_df.iloc[idx][["title","artist","genre","mood","tags","search_query"]].copy()
    res["score"] = sims[idx]
    return res

def predict_emotion_from_face(image_bgr):
    if cnn_model is None or class_names is None:
        return None, None
    crop = detect_and_crop_face(image_bgr)
    x = np.expand_dims(crop, axis=0)
    probs = cnn_model.predict(x, verbose=0)[0]
    pred_id = int(np.argmax(probs))
    return class_names[pred_id], float(np.max(probs))

def predict_emotion_from_text(text):
    # Hybrid: VADER + tiny keyword cues for specific emotions
    analyzer = SentimentIntensityAnalyzer()
    s = analyzer.polarity_scores(text)
    compound = s["compound"]
    text_l = text.lower()

    cues = {
        "angry": ["furious","angry","rage","annoyed","irritated","mad"],
        "fear": ["afraid","scared","terrified","nervous","worried","anxious"],
        "disgust": ["disgust","gross","nasty","revolting","repulsed"],
        "sad": ["sad","depressed","down","unhappy","miserable","blue","cry"],
        "surprise": ["surprised","shocked","astonished","amazed","wow"],
        "happy": ["happy","joyful","glad","excited","delighted","great"],
    }

    # Keyword override
    for emo, kw in cues.items():
        if any(k in text_l for k in kw):
            return emo, 0.9

    if compound >= 0.5:
        return "happy", compound
    elif compound <= -0.6:
        # choose between sad/angry via intensity of "!" etc.
        if text_l.count("!") >= 2 or any(w in text_l for w in ["furious","hate","anger","rage","annoyed"]):
            return "angry", abs(compound)
        return "sad", abs(compound)
    else:
        return "neutral", 1.0 - abs(compound)

st.title("🎵 MoodMate — Emotion → Music")
st.caption("Detect emotion from a face photo or text, then get a mood-aligned playlist.")

tab_img, tab_txt = st.tabs(["📷 From Image", "✍️ From Text"])

with tab_img:
    st.subheader("Upload a face photo")
    img_file = st.file_uploader("Image file", type=["jpg","jpeg","png"])
    if img_file is not None:
        image_bytes = img_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        emo, conf = predict_emotion_from_face(bgr)
        if emo is None:
            st.warning("Model not found. Please train the CNN first.")
        else:
            st.success(f"Predicted emotion: **{emo}** (confidence {conf:.2f})")
            recs = recommend_for_emotion(emo, top_k=10)
            if recs.empty:
                st.info("Recommender index missing. Run the recommender builder script.")
            else:
                st.dataframe(recs)

with tab_txt:
    st.subheader("Describe how you feel")
    txt = st.text_area("Type a sentence or two...", "")
    if st.button("Analyze & Recommend", type="primary"):
        if not txt.strip():
            st.warning("Please enter some text.")
        else:
            emo, conf = predict_emotion_from_text(txt)
            st.success(f"Detected emotion: **{emo}** (confidence {conf:.2f})")
            recs = recommend_for_emotion(emo, top_k=10)
            if recs.empty:
                st.info("Recommender index missing. Run the recommender builder script.")
            else:
                st.dataframe(recs)
