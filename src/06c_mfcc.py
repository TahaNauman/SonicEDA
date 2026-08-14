"""Step 6C: MFCCs - a compact numerical description of timbre."""

from config import AUDIO_FILE, ROOT, SR, FRAME, HOP
from pathlib import Path

import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np


def load_audio(path: Path, sr: int = SR):
    """Load the song as mono samples."""
    return librosa.load(path, sr=sr)


def extract_mfccs(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    frame_length: int = FRAME,
    hop_length: int = HOP,
) -> np.ndarray:
    """Extract MFCC coefficients for each frame of the song."""
    return librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=frame_length,
        hop_length=hop_length,
    )


def main() -> None:
    y, sr = load_audio(AUDIO_FILE)

    mfccs = extract_mfccs(y, sr)

    print("MFCC ANALYSIS")
    print("=" * 50)

    print(f"Audio duration       : {len(y) / sr:.2f} seconds")
    print(f"Sample rate          : {sr} Hz")
    print(f"Number of MFCCs      : {mfccs.shape[0]}")
    print(f"Number of time frames: {mfccs.shape[1]}")

    print("\nMFCC matrix shape:")
    print(f"  {mfccs.shape[0]} coefficients × {mfccs.shape[1]} frames")

    print("\nFirst frame:")
    for i, value in enumerate(mfccs[:, 0], start=1):
        print(f"  MFCC {i:2d}: {value:8.2f}")

    print("\nAverage value of each coefficient:")
    for i, value in enumerate(mfccs.mean(axis=1), start=1):
        print(f"  MFCC {i:2d}: {value:8.2f}")

    # Convert frame numbers into actual time
    times = librosa.frames_to_time(
        np.arange(mfccs.shape[1]),
        sr=sr,
        hop_length=HOP,
    )

    # Plot all 13 MFCCs over time
    plt.figure(figsize=(12, 6))

    librosa.display.specshow(
        mfccs,
        x_axis="time",
        sr=sr,
        hop_length=HOP,
        cmap="magma",
    )

    plt.colorbar(label="MFCC value")
    plt.xlabel("Time (seconds)")
    plt.ylabel("MFCC coefficient")
    plt.title("MFCCs — Timbre Characteristics Over Time")

    plt.tight_layout()

    out = ROOT / "plots" / "06c_mfcc.png"
    plt.savefig(out, dpi=100)

    plt.figure(figsize=(12, 6))

    times = librosa.frames_to_time(
        np.arange(mfccs.shape[1]),
        sr=sr,
        hop_length=HOP,
    )

    for i in range(5):
        plt.plot(times, mfccs[i], label=f"MFCC {i + 1}")

    plt.xlabel("Time (seconds)")
    plt.ylabel("MFCC value")
    plt.title("First 5 MFCC Coefficients Over Time")
    plt.legend()

    plt.tight_layout()

    out = ROOT / "plots" / "06c_mfcc_lines.png"
    plt.savefig(out, dpi=100)

    print(f"saved {out}")

if __name__ == "__main__":
    main()