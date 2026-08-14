from config import AUDIO_FILE, ROOT, SR, FRAME, HOP

import matplotlib.pyplot as plt
import librosa
import numpy as np


def main():
    y, sr = librosa.load(AUDIO_FILE, sr=SR)

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=FRAME,
        hop_length=HOP,
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr,
        n_fft=FRAME,
        hop_length=HOP,
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr,
        n_fft=FRAME,
        hop_length=HOP,
    )[0]

    print("SPECTRAL FEATURES\n")

    print(f"Centroid:")
    print(f"  mean : {centroid.mean():.0f} Hz")
    print(f"  min  : {centroid.min():.0f} Hz")
    print(f"  max  : {centroid.max():.0f} Hz")

    print()

    print(f"Rolloff:")
    print(f"  mean : {rolloff.mean():.0f} Hz")
    print(f"  min  : {rolloff.min():.0f} Hz")
    print(f"  max  : {rolloff.max():.0f} Hz")

    print()

    print(f"Bandwidth:")
    print(f"  mean : {bandwidth.mean():.0f} Hz")
    print(f"  min  : {bandwidth.min():.0f} Hz")
    print(f"  max  : {bandwidth.max():.0f} Hz")

        # --- plots -----------------------------------------------------------

    times = librosa.frames_to_time(
        np.arange(len(centroid)),
        sr=sr,
        hop_length=HOP
    )

    # 1. Spectral centroid
    plt.figure(figsize=(12, 4))
    plt.plot(times, centroid, linewidth=0.8)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectral Centroid — Frequency Brightness Over Time")
    plt.tight_layout()

    out = ROOT / "plots" / "06b_spectral_centroid.png"
    plt.savefig(out, dpi=100)
    plt.show()


    # 2. Spectral rolloff
    plt.figure(figsize=(12, 4))
    plt.plot(times, rolloff, linewidth=0.8)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectral Rolloff — Frequency Below Which 85% of Energy Lies")
    plt.tight_layout()

    out = ROOT / "plots" / "06b_spectral_rolloff.png"
    plt.savefig(out, dpi=100)
    plt.show()


    # 3. Spectral bandwidth
    plt.figure(figsize=(12, 4))
    plt.plot(times, bandwidth, linewidth=0.8)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Bandwidth (Hz)")
    plt.title("Spectral Bandwidth — Spread of Frequency Content")
    plt.tight_layout()

    out = ROOT / "plots" / "06b_spectral_bandwidth.png"
    plt.savefig(out, dpi=100)
    plt.show()


if __name__ == "__main__":
    main()