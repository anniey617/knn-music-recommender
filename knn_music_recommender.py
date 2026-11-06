"""
KNN Music Recommender
=====================

Recommends songs similar to a given track using k-Nearest Neighbours over
Spotify audio features (danceability, energy, tempo, and so on).

How it works:
  1. A dataset of tracks and their audio features is loaded from a CSV.
  2. A k-NN model is fit over those features using Manhattan distance.
  3. For an input song, its audio features are looked up via the Spotify Web
     API, and the model returns the closest matching tracks.

Setup:
  Credentials are read from environment variables so they are never committed
  to source control:

      export SPOTIFY_CLIENT_ID="your_client_id"
      export SPOTIFY_CLIENT_SECRET="your_client_secret"
      python knn_music_recommender.py

Author: Annie Yang
"""

import os
import base64

import requests
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Audio features used as the model's input dimensions (lowercase = Spotify API
# field names). The dataset CSV stores the same fields capitalised.
AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness", "valence", "tempo",
]

DATASET_PATH = "tracks.csv"


def get_spotify_token(client_id, client_secret):
    """Request a client-credentials access token from the Spotify API."""
    auth_header = base64.b64encode(
        f"{client_id}:{client_secret}".encode()
    ).decode()

    response = requests.post(
        "https://accounts.spotify.com/api/token",
        headers={"Authorization": f"Basic {auth_header}"},
        data={"grant_type": "client_credentials"},
    )
    response.raise_for_status()
    return response.json()["access_token"]


def search_song(token, song_title, artist_name):
    """Return the Spotify track ID for a title/artist pair, or None if not found."""
    response = requests.get(
        "https://api.spotify.com/v1/search",
        headers={"Authorization": f"Bearer {token}"},
        params={
            "q": f"track:{song_title} artist:{artist_name}",
            "type": "track",   # search tracks only, not albums or artists
            "limit": 1,
        },
    )
    response.raise_for_status()

    items = response.json().get("tracks", {}).get("items", [])
    return items[0]["id"] if items else None


def get_audio_features(token, song_id):
    """Return the audio-features dict for a track, or None if unavailable."""
    response = requests.get(
        f"https://api.spotify.com/v1/audio-features/{song_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    if response.status_code != 200:
        return None
    return response.json()


def train_knn(data, n_neighbors=10):
    """Fit a k-NN model over the dataset's audio features.

    Returns the fitted model and the cleaned dataframe (rows with missing
    features dropped) so indices stay aligned with the model.
    """
    columns = [feature.capitalize() for feature in AUDIO_FEATURES]
    data = data.dropna(subset=columns)

    model = NearestNeighbors(n_neighbors=n_neighbors, p=1)  # p=1 -> Manhattan
    model.fit(data[columns].values)
    return model, data


def get_similar_songs(model, data, input_features):
    """Return the tracks whose audio features are nearest to the input song."""
    query = [input_features[feature] for feature in AUDIO_FEATURES]
    _, indices = model.kneighbors([query])

    display_columns = ["Track Name", "Artist Name(s)"] + [
        feature.capitalize() for feature in AUDIO_FEATURES
    ]
    return data.iloc[indices[0]][display_columns]


def main():
    try:
        client_id = os.environ["SPOTIFY_CLIENT_ID"]
        client_secret = os.environ["SPOTIFY_CLIENT_SECRET"]
    except KeyError:
        raise SystemExit(
            "Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment "
            "variables before running."
        )

    token = get_spotify_token(client_id, client_secret)

    data = pd.read_csv(DATASET_PATH)
    model, data = train_knn(data)

    song_title = input("Enter song title: ")
    artist_name = input("Enter song artist (main one): ")

    song_id = search_song(token, song_title, artist_name)
    features = get_audio_features(token, song_id) if song_id else None

    if not features:
        print("Could not retrieve audio features for that song.")
        return

    print(f"\nSongs similar to '{song_title}' by {artist_name}:\n")
    print(get_similar_songs(model, data, features).to_string(index=False))


if __name__ == "__main__":
    main()
