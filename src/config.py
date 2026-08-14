from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO_FILE = ROOT / "data" / "Pokemon.mp3"

SR = 22050        # sample rate
FRAME = 2048      # frame_length (samples per window)
HOP = 1024        # hop_length (steps between windows)

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]