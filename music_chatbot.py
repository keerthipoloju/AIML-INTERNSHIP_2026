import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import re

# ---------------------- Spotify API Auth ---------------------- #
# Replace with your credentials
CLIENT_ID = "2104b309ad774871ba1e2af8ba233c5b"
CLIENT_SECRET = "8874da9c130641e9ab6dc5996890eab0"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET
))

# ---------------------- Helper Function ---------------------- #
def get_songs_by_artist_spotify(artist_name: str, limit=5):
    """Fetch top tracks by artist from Spotify"""
    results = sp.search(q=f"artist:{artist_name}", type="artist", limit=1)
    if not results["artists"]["items"]:
        return []

    artist_id = results["artists"]["items"][0]["id"]

    # Get top tracks
    top_tracks = sp.artist_top_tracks(artist_id, country="IN")  # change country if needed
    songs = []
    for track in top_tracks["tracks"][:limit]:
        songs.append({
            "title": track["name"],
            "url": track["external_urls"]["spotify"],
            "album": track["album"]["name"],
            "preview": track["preview_url"]
        })
    return songs

# ---------------------- Streamlit UI ---------------------- #
st.set_page_config(page_title="🎤 Artist Chatbot", page_icon="🎶")

st.title("🎤 Artist Chatbot (Spotify Edition)")
st.write("Ask me about your favorite artist and I’ll fetch their top songs from Spotify! 🎶")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_query = st.chat_input("💬 Example: Show me songs by Arijit Singh")

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # Updated regex to handle "by/of" queries or just a name
    match = re.search(r"(?:by|of)\s+(.+)|(.+)", user_query, re.IGNORECASE)
    
    if match:
        # The captured group might be in group 1 or group 2
        artist_name = match.group(1) or match.group(2)
        artist_name = artist_name.strip()
        
        results = get_songs_by_artist_spotify(artist_name)
        
        if results:
            bot_response = f"Here are some top songs by **{artist_name}**:\n\n"
            for song in results:
                bot_response += f"- 🎵 [{song['title']}]({song['url']}) (Album: {song['album']})\n"
                if song["preview"]:
                    bot_response += f"  🔊 [Preview]({song['preview']})\n"
        else:
            bot_response = f"Sorry, I couldn’t find any songs by **{artist_name}**."
    else:
        # This case is less likely with the new regex but good to have
        bot_response = "Try asking like: *Show me songs by Arijit Singh* or just *Arijit Singh* 🎵"

    st.session_state.chat_history.append({"role": "bot", "content": bot_response})

# Display chat history
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"🧑 **You:** {chat['content']}")
    else:
        st.markdown(f"🤖 **Bot:** {chat['content']}")
