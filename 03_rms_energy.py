"""Step 3: RMS energy - loudness as a curve over time, not one flat number."""

from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt

AUDIO_FILE = Path("data/Pokemon.mp3")
FRAME = 2048   # samples per window (~93 ms at 22050 Hz)
HOP = 1024     # steps between windows (50% overlap)

def load(path: Path, sr: int = 22050):
    return librosa.load(path, sr=sr)

def rms_by_hand(y, frame_length: int, hop_length: int) -> np.ndarray:
    """RMS per frame, written out so the math is visible."""
    frames = np.lib.stride_tricks.sliding_window_view(
        y, frame_length
    )[::hop_length]               # every hop-th window
    return np.sqrt(np.mean(frames**2, axis=1))

def main() -> None:
    y, sr = load(AUDIO_FILE)

    mine = rms_by_hand(y, FRAME, HOP)
    librosas = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP, center=False)[0]

    n = len(mine)  # matching windows once both start at sample 0
    print(f"frames computed      : {len(mine):,}")
    print(f"approx ms per frame  : {HOP / sr * 1000:.0f} (one RMS value per {HOP/sr*1000:.0f} ms of song)")
    print(f"my RMS vs librosa RMS: max |diff| = {np.abs(mine[:n] - librosas[:n]).max():.2e}  <- same math")
    print(f"loudest frame at {np.argmax(mine) * HOP / sr:.1f}s (RMS {mine.max():.3f})")
    print(f"quietest frame at {np.argmin(mine) * HOP / sr:.1f}s (RMS {mine.min():.3f})")

    print("\nStructure map (rough, one block per ~3s of song):")
    blocks = 64
    per_block = len(mine) / blocks
    for b in range(blocks):
        seg = mine[int(b * per_block): int((b + 1) * per_block)]
        bar = "#" * int(round(seg.mean() / mine.max() * 50))
        print(f"  {b * per_block * HOP / sr:5.0f}s  {bar}")
    sample_times = np.arange(len(y)) / sr
    times = librosa.frames_to_time(np.arange(len(mine)), sr=sr, hop_length=HOP)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(sample_times[::20], y[::20], linewidth=0.3)
    axes[0].set_title("Waveform")
    axes[1].plot(times, librosas, linewidth=1.0)
    axes[1].set_title("RMS energy over time")
    axes[1].set_xlabel("time (seconds)")
    plt.tight_layout()
    out = "plots/03_rms_energy.png"
    plt.savefig(out, dpi=100)
    print(f"\nsaved {out}")

if __name__ == "__main__":
    main()