"""Step 6D: Chroma - pitch-class energy over time."""

from config import AUDIO_FILE, ROOT, SR, HOP, PITCH_NAMES
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def load_audio(path: Path, sr: int = SR):
    """Load the song as mono audio."""
    return librosa.load(path, sr=sr)


def extract_chroma(
    y: np.ndarray,
    sr: int,
    hop_length: int = HOP,
) -> np.ndarray:
    """Calculate the strength of each of the 12 pitch classes over time."""
    return librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
    )


def main() -> None:
    y, sr = load_audio(AUDIO_FILE)

    chroma = extract_chroma(y, sr)

    print("CHROMA ANALYSIS")
    print("=" * 50)

    print(f"Audio duration       : {len(y) / sr:.2f} seconds")
    print(f"Sample rate          : {sr} Hz")
    print(f"Number of pitch classes: {chroma.shape[0]}")
    print(f"Number of time frames: {chroma.shape[1]}")

    print("\nPitch classes:")
    for i, name in enumerate(PITCH_NAMES):
        print(f"  {i:2d}: {name}")

    # Average strength of each pitch class across the whole song.
    mean_chroma = chroma.mean(axis=1)

    print("\nAverage pitch-class strength:")
    for name, value in zip(PITCH_NAMES, mean_chroma):
        print(f"  {name:2s}: {value:.3f}")

    strongest = np.argsort(mean_chroma)[-3:][::-1]

    print("\nThree strongest pitch classes:")
    for i in strongest:
        print(f"  {PITCH_NAMES[i]}: {mean_chroma[i]:.3f}")

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    plt.figure(figsize=(12, 6))

    librosa.display.specshow(
        chroma,
        x_axis="time",
        y_axis="chroma",
        sr=sr,
        hop_length=HOP,
        cmap="magma",
    )

    plt.colorbar(label="pitch-class strength")

    plt.title("Chroma — Pitch Classes Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Pitch class")

    plt.tight_layout()

    out = ROOT / "plots" / "06d_chroma.png"
    plt.savefig(out, dpi=100)

    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()