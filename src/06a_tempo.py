"""Step 6A: tempo and beat tracking - the pulse of the song."""

from config import AUDIO_FILE, ROOT, SR, FRAME, HOP

import matplotlib.pyplot as plt

import librosa
import numpy as np


def main():
    y, sr = librosa.load(AUDIO_FILE, sr=SR)

    tempo, beat_frames = librosa.beat.beat_track(
        y=y,
        sr=sr
    )

    tempo = float(np.asarray(tempo).mean())

    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=sr
    )

    print(f"Tempo: {tempo:.1f} BPM")
    print(f"Number of detected beats: {len(beat_times)}")

    print("\nFirst 10 detected beats:")

    for i, time in enumerate(beat_times[:10], start=1):
        print(f"  Beat {i:2}: {time:.2f} seconds")


    rms = librosa.feature.rms(
        y=y,
        frame_length=FRAME,
        hop_length=HOP,
        center=False
    )[0]

    rms_times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sr,
        hop_length=HOP
    )

    plt.figure(figsize=(14, 4))

    plt.plot(
        rms_times,
        rms,
        linewidth=0.8,
        label="RMS energy"
    )

    plt.vlines(
        beat_times,
        0,
        rms.max(),
        linewidth=0.8,
        alpha=0.6,
        label="Detected beats"
    )

    plt.xlim(0, 10)

    plt.xlabel("Time (seconds)")
    plt.ylabel("RMS energy")
    plt.title("Detected Beats vs RMS Energy — First 10 Seconds")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        ROOT / "plots" / "06a_tempo_beats.png",
        dpi=100
    )

if __name__ == "__main__":
    main()