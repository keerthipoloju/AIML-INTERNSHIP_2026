# 🎵 MoodMate — Emotion → Music Recommender

## 📌 Overview
MoodMate is a Streamlit-based application that detects emotions from **face images** or **text input** and recommends music playlists aligned with the detected mood.  
It combines:
- A CNN model for facial emotion recognition.
- VADER sentiment analysis + keyword cues for text emotion detection.
- A recommender system using TF-IDF vectorization and a song index.

---

## 🛠️ Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
