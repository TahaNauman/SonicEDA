"""Step 6: feature extraction - distilling the song into meaningful numbers."""

from config import AUDIO_FILE, ROOT, SR, FRAME, HOP
import librosa
import numpy as np
import matplotlib.pyplot as plt

def report(name: str, value: str) -> None:
    print(f"{name:>22} : {value}")

def main() -> None:
    y, sr = librosa.load(AUDIO_FILE, sr=SR)

    print("FEATURES EXTRACTED FROM Pokemon.mp3\n")

    # --- tempo and beats -------------------------------------------------
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.asarray(tempo).mean())
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    report("tempo (BPM)", f"{float(tempo):.1f} beats per minute")
    report("beats", f"{len(beat_times)} pulse points over {y.shape[0] / sr:.0f}s "
                    f"(first at {beat_times[0]:.2f}s, last at {beat_times[-1]:.2f}s)")

    # --- spectral features (built on the step-5 spectrogram) -------------
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=FRAME, hop_length=HOP)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=FRAME, hop_length=HOP)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=FRAME, hop_length=HOP)[0]
    report("spectral centroid", f"{centroid.mean():6.0f} Hz avg brightness "
                                f"(min {centroid.min():.0f}, max {centroid.max():.0f})")
    report("spectral rolloff", f"{rolloff.mean():6.0f} Hz avg - 85% of energy is below this")
    report("spectral bandwidth", f"{bandwidth.mean():6.0f} Hz avg spread around the centroid")

    # --- MFCCs (timbre fingerprint) -------------------------------------
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=FRAME, hop_length=HOP)
    report("MFCC", f"13 coefficients x {mfccs.shape[1]} frames; "
                   f"per-frame mean: " + ", ".join(f"{m:.1f}" for m in mfccs.mean(axis=1)[:6]) + " ...")

    # --- chroma (12 pitch classes) --------------------------------------
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    strongest = np.argsort(chroma.mean(axis=1))[-3:][::-1]
    report("chroma", "strongest pitch classes: "
                     + ", ".join(f"{names[i]} ({chroma.mean(axis=1)[i]:.2f})" for i in strongest)
                     + "  <- harmony/tonal emphasis")

    # --- plots -----------------------------------------------------------
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP, center=False)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].plot(times, rms, linewidth=0.8)
    axes[0, 0].vlines(beat_times, 0, rms.max(), color="r", linewidth=0.8, alpha=0.6)
    axes[0, 0].set_title("Beats (red) on the loudness envelope")
    axes[0, 0].set_xlabel("time (s)")

    axes[0, 1].plot(times, centroid[: len(times)], linewidth=0.8, label="centroid")
    axes[0, 1].plot(times, rolloff[: len(times)], linewidth=0.8, label="rolloff")
    axes[0, 1].set_title("Spectral centroid & rolloff (brightness)")
    axes[0, 1].legend()

    img = librosa.display.specshow(mfccs, sr=sr, hop_length=HOP, x_axis="time", ax=axes[1, 0])
    axes[1, 0].set_title("MFCCs (timbre fingerprint)")
    plt.colorbar(img, ax=axes[1, 0])

    img = librosa.display.specshow(chroma, sr=sr, hop_length=HOP, y_axis="chroma", x_axis="time", ax=axes[1, 1])
    axes[1, 1].set_title("Chroma (12 pitch classes over time)")
    plt.colorbar(img, ax=axes[1, 1])

    plt.tight_layout()
    out = ROOT / "plots" / "06_features.png"
    plt.savefig(out, dpi=100)
    print(f"\nsaved {out}")

if __name__ == "__main__":
    main()