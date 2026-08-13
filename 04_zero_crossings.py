"""Step 4: zero crossings - how often does the signal flip sign? A rough brightness meter."""

from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt

AUDIO_FILE = Path("data/Pokemon.mp3")
FRAME = 2048
HOP = 1024

def load(path: Path, sr: int = 22050):
    return librosa.load(path, sr=sr)

def zcr_by_hand(y) -> float:
    """Fraction of samples that flip sign versus their neighbor."""
    flips = (y[:-1] > 0) != (y[1:] > 0)   # True where sign changes
    return flips.mean()                   # crossings per sample

def zcr_frames_by_hand(y, frame_length: int, hop_length: int) -> np.ndarray:
    """Zero-crossing rate per frame, same framing trick as step 3."""
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_length)[::hop_length]
    signs = frames > 0
    flips = (signs[:, :-1] != signs[:, 1:]).mean(axis=1)
    return flips

def main() -> None:
    y, sr = load(AUDIO_FILE)

    mine = zcr_by_hand(y)
    librosas = librosa.feature.zero_crossing_rate(y)[0].mean()
    print(f"overall zero-crossing rate (per sample) : {mine:.5f}")
    print(f"librosa's rate                          : {librosas:.5f}")
    print(f"  that's about {mine * sr:,.0f} sign flips per second")

    fimine = zcr_frames_by_hand(y, FRAME, HOP)
    filib = librosa.feature.zero_crossing_rate(
        y, frame_length=FRAME, hop_length=HOP, center=False, threshold=0.0
    )[0]
    n = min(len(fimine), len(filib))
    print(f"per-frame rate: max |diff| vs librosa = {np.abs(fimine[:n] - filib[:n]).max():.2e}  (tiny residual = frame-edge handling)")
    print(f"Highest ZCR at {np.argmax(fimine) * HOP / sr:.1f}s (rate {fimine.max():.3f})")
    print(f"Lowest ZCR at {np.argmin(fimine) * HOP / sr:.1f}s (rate {fimine.min():.3f})")

    times = librosa.frames_to_time(np.arange(len(fimine)), sr=sr, hop_length=HOP)
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP, center=False)[0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    axes[0].plot(times, fimine, linewidth=0.8)
    axes[0].set_title("Zero-Crossing Rate over Time")
    axes[0].set_ylabel("crossings per sample")
    axes[1].scatter(rms[:n], fimine[:n], s=3, alpha=0.4)
    axes[1].set_title("RMS vs Zero-Crossing Rate")
    axes[1].set_xlabel("RMS")
    axes[1].set_ylabel("Zero-Crossing Rate")
    plt.tight_layout()
    out = "plots/04_zero_crossings.png"
    plt.savefig(out, dpi=100)
    print(f"\nsaved {out}")

if __name__ == "__main__":
    main()