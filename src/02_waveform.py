"""Step 2: the raw samples - what does the sound look like as numbers?"""

from config import AUDIO_FILE, ROOT, SR
import librosa
import matplotlib.pyplot as plt

def view_samples(y, start: int, count: int) -> None:
    """Print raw values to see the actual numbers the sound is made of."""
    print(f"  samples {start}..{start + count - 1} (each one is the amplitude at a moment):")
    for i in range(start, start + count):
        print(f"    y[{i}] = {y[i]:+.4f}")

def plot_waveform(y, sr: int, out_file: str) -> None:
    """Graph all 4.3M samples: time on x-axis, amplitude (loudness) on y-axis."""
    seconds = librosa.times_like(y, sr=sr)
    plt.figure(figsize=(12, 3))
    plt.plot(seconds, y, linewidth=0.3)
    plt.xlabel("time (seconds)")
    plt.ylabel("amplitude")
    plt.title("Waveform: amplitude of the sound over time")
    plt.tight_layout()
    plt.savefig(out_file, dpi=100)
    print(f"  saved plot to {out_file}")

def main() -> None:
    y, sr = librosa.load(AUDIO_FILE, sr=SR)

    print(f"decoded: {len(y):,} samples at {sr} Hz")
    print(f"loudest sample : {y.max():+.4f}")
    print(f"quietest sample: {y.min():+.4f}")
    print(f"overall loudness (RMS) of entire song: {librosa.feature.rms(y=y).mean():+.4f}")
    print()

    print("A close-up of the first second of audio (samples 0..15):")
    view_samples(y, 0, 16)

    plot_waveform(y, sr, ROOT / "plots" / "02_waveform.png")

    print("\nDone. Open plots/02_waveform.png to see the sound as a picture.")

if __name__ == "__main__":
    main()