"""Track registry helpers (no Isaac imports, usable before the app starts). Tracks live in rl/tracks/<slug>/{track_col.usda,centerline.npy,meta.json}."""
import json, os, sys

TRACKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracks")


def list_tracks():
    return sorted(d for d in os.listdir(TRACKS) if os.path.exists(os.path.join(TRACKS, d, "centerline.npy")))


def track_dir(name):
    if name not in list_tracks():
        raise ValueError(f"unknown track '{name}'; available: {', '.join(list_tracks())}")
    return os.path.join(TRACKS, name)


def track_meta(name):
    return json.load(open(os.path.join(track_dir(name), "meta.json")))


def add_track_arg(p):
    """--track NAME (default icra25); '--track list' prints the names and exits. Call after all args are added, before AppLauncher."""
    p.add_argument("--track", default="icra25", help="track name, 'list' to print the available ones")


def check_track(name):
    if name == "list":
        print("available tracks:", ", ".join(list_tracks())); sys.exit(0)
    try: track_dir(name)
    except ValueError as e: sys.exit(str(e))
