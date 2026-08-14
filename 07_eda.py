"""Step 7: EDA - aligning all features into one table and asking it questions."""

from pathlib import Path
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

AUDIO_FILE = Path("data/Pokemon.mp3")
SR = 22050
FRAME = 2048
HOP = 1024
CHROMA_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def build_frame_table(y, sr: int) -> pd.DataFrame:
    """Every feature aligned to the same frame grid = one table row per moment."""
    hop = HOP
    n_frames = (len(y) - FRAME) // hop + 1

    cols = {
        "time_s": np.arange(n_frames) * hop / sr,
        "rms": librosa.feature.rms(y=y, frame_length=FRAME, hop_length=hop, center=False)[0, :n_frames],
        "zcr": librosa.feature.zero_crossing_rate(y, frame_length=FRAME, hop_length=hop, center=False, threshold=0.0)[0, :n_frames],
        "centroid_hz": librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=FRAME, hop_length=hop)[0, :n_frames],
        "rolloff_hz": librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=FRAME, hop_length=hop)[0, :n_frames],
        "bandwidth_hz": librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=FRAME, hop_length=hop)[0, :n_frames],
        "mfcc0": librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=FRAME, hop_length=hop)[0, :n_frames],
    }

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    cols["chroma"] = [CHROMA_NAMES[np.argmax(c)] for c in chroma.T[:n_frames]]
    return pd.DataFrame(cols)

def main() -> None:
    y, sr = librosa.load(AUDIO_FILE, sr=SR)
    df = build_frame_table(y, sr)
    print(f"DATA TABLE: {df.shape[0]:,} rows (one per ~46ms moment) x {df.shape[1]} columns (features)")
    print(df.head().to_string(index=False))

    print("\n--- summary statistics (describe) -----------------------------")
    print(df[["rms", "zcr", "centroid_hz", "rolloff_hz", "bandwidth_hz"]].describe().round(3).to_string())

    print("\n--- correlations between features -----------------------------")
    corr = df[["rms", "zcr", "centroid_hz", "rolloff_hz", "bandwidth_hz", "mfcc0"]].corr()
    pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().dropna().sort_values(ascending=False)
    for (a, b), v in pairs.items():
        print(f"  {a:>13} ~ {b:>13} : {v:+.3f}")

    print("\n--- features change over time: three sections -----------------")
    df["second"] = df["time_s"].astype(int)
    for name, lo, hi in [("intro 0-30s", 0, 30), ("mid 100-130s", 100, 130), ("climax 130-160s", 130, 160)]:
        seg = df[(df["time_s"] >= lo) & (df["time_s"] < hi)]
        print(f"  {name:>16}: rms {seg['rms'].mean():.3f}  centroid {seg['centroid_hz'].mean():5.0f} Hz  "
              f"top pitch {seg['chroma'].mode().iloc[0]}")

    print("\n--- per-second profile (first 12 s of the song) ---------------")
    per_sec = df.groupby("second")[["rms", "centroid_hz"]].mean().head(12)
    print(per_sec.round(3).to_string())

    # --- plots -----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].hist(df["rms"], bins=60)
    axes[0, 0].set_title("RMS distribution (loudness)")
    axes[0, 1].hist(df["centroid_hz"], bins=60)
    axes[0, 1].set_title("Centroid distribution (brightness)")
    sns.heatmap(corr, annot=True, fmt="+.2f", ax=axes[1, 0])
    axes[1, 0].set_title("Feature correlations")
    df[["rms", "zcr", "centroid_hz", "rolloff_hz", "bandwidth_hz"]].plot.box(ax=axes[1, 1], rot=20)
    axes[1, 1].set_title("Feature distributions (boxplots)")
    plt.tight_layout()
    out = "plots/07_eda.png"
    plt.savefig(out, dpi=100)
    print(f"\nsaved {out}")

if __name__ == "__main__":
    main()