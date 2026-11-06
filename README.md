# KNN Music Recommender

A song recommender that suggests tracks similar to a given song using the
k-Nearest Neighbours algorithm over Spotify audio features (danceability,
energy, tempo, and so on).

## How it works
1. Loads a dataset of tracks and their audio features from a CSV.
2. Fits a k-NN model over those features using Manhattan distance.
3. Looks up an input song's audio features via the Spotify Web API and returns
   the closest matching tracks.

## Running it
Credentials are read from environment variables so they are never committed to
source control:

```bash
export SPOTIFY_CLIENT_ID="your_client_id"
export SPOTIFY_CLIENT_SECRET="your_client_secret"
python knn_music_recommender.py
```

A `tracks.csv` file of songs and their audio features is expected in the same
directory.

## Note
Spotify has since restricted its `/audio-features` endpoint, so the live
input-song lookup depends on API access. The k-NN recommender itself runs over
the dataset independently of that.

Author: Annie Yang
