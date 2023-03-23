"""
This script connects to the Spotify API via the spotipy library to download the album covers of your saved songs.
Follow the instructions in the spotipy documentation to obtain a client ID and client secret for the Spotify API.
As specified in section "Authorization Code Flow", the environment variables SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET
and SPOTIPY_REDIRECT_URI must be set to enable access to the Spotify API.
"""
import os
import urllib.request
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
import spotipy
from spotipy.oauth2 import SpotifyOAuth


def download_album_covers_for_saved_tracks(target_dir: str):
    all_saved_tracks = get_all_saved_tracks()
    unique_albums = extract_unique_albums(all_saved_tracks)
    for album in tqdm(unique_albums, desc=f"Downloading album covers to {target_dir}"):
        download_album_cover(album, target_dir)


def get_all_saved_tracks():
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope="user-library-read"))
    num_saved_tracks = sp.current_user_saved_tracks(limit=1, offset=0)['total']
    tracks_per_page = 50
    num_pages = int(np.ceil(num_saved_tracks / tracks_per_page))
    for i in range(num_pages):
        results = sp.current_user_saved_tracks(limit=tracks_per_page, offset=i * tracks_per_page)
        yield from [item['track'] for item in results['items']]


def extract_unique_albums(tracks):
    return {track['album']['id']: track['album'] for track in tracks}.values()


def download_album_cover(album: dict, target_dir: str):
    os.makedirs(target_dir, exist_ok=True)
    if not os.path.isfile(local_target_path := f"{target_dir}/{album['id']}.jpg"):
        urllib.request.urlretrieve(album['images'][0]['url'], local_target_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("target_path", help="Path to target directory where the album covers should be saved")
    args = parser.parse_args()
    download_album_covers_for_saved_tracks(args.target_path)
