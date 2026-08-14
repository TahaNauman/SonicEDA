"""Step 2a: does the sample rate choice (22050 vs 44100) change what we get?"""

from config import AUDIO_FILE, ROOT
from pathlib import Path
import time
import librosa
import matplotlib.pyplot as plt

SLICE = (30.0, 32.0)  # compare the same 2 seconds at both rates

def load(path: Path, sr=None):
    """sr=None -> librosa keeps the file's native sample rate (44,100 for mp3 CD-era)."""
    t0 = time.perf_counter()
    y, sr = librosa.load(path, sr=sr)
    elapsed = time.perf_counter() - t0
    return y, sr, elapsed

def main() -> None:
    y_slow, sr_slow, t_slow = load(AUDIO_FILE, sr=22050)
    y_native, sr_native, t_native = load(AUDIO_FILE, sr=None)

    print(f"{'':>25} {'analysis rate (22050)':>24} {'native rate (44100)':>24}")
    print(f"{'sample rate (Hz)':>25} {sr_slow:>24} {sr_native:>24}")
    print(f"{'number of samples':>25} {len(y_slow):>24,} {len(y_native):>24,}")
    print(f"{'load time (s)':>25} {t_slow:>24.2f} {t_native:>24.2f}")

    rms_slow = librosa.feature.rms(y=y_slow).mean()
    rms_native = librosa.feature.rms(y=y_native).mean()
    print(f"{'overall RMS':>25} {rms_slow:>24.4f} {rms_native:>24.4f}")
    print("\nLesson: half the samples at 22050 -- and the loudness it measures is the same.")
    print("(load time here is a wash: decoding+resampling can cost more than the raw decode.)")

    start = int(SLICE[0] * sr_slow)
    end = int(SLICE[1] * sr_slow)
    start_n = int(SLICE[0] * sr_native)
    end_n = int(SLICE[1] * sr_native)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].plot(librosa.times_like(y_slow[start:end], sr=sr_slow), y_slow[start:end], linewidth=0.5)
    axes[0].set_title(f"22050 Hz -- {SLICE[0]}..{SLICE[1]}s")
    axes[1].plot(librosa.times_like(y_native[start_n:end_n], sr=sr_native), y_native[start_n:end_n], linewidth=0.5)
    axes[1].set_title(f"44100 Hz -- {SLICE[0]}..{SLICE[1]}s")
    axes[1].set_xlabel("time (seconds)")
    fig.suptitle("The same 2 seconds at both sample rates -- look identical")
    plt.tight_layout()
    out = ROOT / "plots" / "02a_waveforms.png"
    plt.savefig(out, dpi=100)
    print(f"\nsaved {out}")

if __name__ == "__main__":
    main()