"""Step 5: the spectrogram - at each moment, how loud is each frequency?"""

from config import AUDIO_FILE, ROOT, FRAME, HOP, SR
import librosa
import numpy as np
import matplotlib.pyplot as plt

def synth(freq_hz: float, sr: int, seconds: float) -> np.ndarray:
    """A pure tone: a sine wave that completes freq_hz oscillations per second."""
    t = np.arange(int(sr * seconds)) / sr
    return np.sin(2 * np.pi * freq_hz * t)

def match_strength(signal: np.ndarray, tone: np.ndarray) -> float:
    """How well 'tone' is hiding inside 'signal' (correlation)."""
    return float(np.abs(np.dot(signal, tone)) / len(signal))

def demo_intuition() -> None:
    """Fourier's core idea, no FFT math: correlate against pure tones."""
    print("PART 1 - the Fourier intuition (pure Python, no FFT)")
    a = synth(440.0, SR, 0.1)   # A4
    b = synth(880.0, SR, 0.1)   # A5, one octave up
    mixed = a + b               # the song: many tones at once

    print(f"  tone 440 Hz alone: a clean wave you recognize as 'A'")
    print(f"  tone 880 Hz alone: same, one octave higher")
    print(f"  mixed: a jagged wave that is neither -- but contains BOTH")
    print()
    print("  Now the trick - correlate the mixed signal against each candidate tone:")
    for name, tone in [("440 Hz (real)", a), ("880 Hz (real)", b), ("500 Hz (not in the mix)", synth(500.0, SR, 0.1))]:
        print(f"    {name:>20} : match = {match_strength(mixed, tone):.4f}")
    print("  High match = that tone IS in the sound. That correlation IS the Fourier transform.\n")

def loudest_frequencies_at(stft_db, sr: int, hop: int, frame_length: int, time_s: float) -> None:
    """Reality check: what is the song actually made of at one moment?"""
    t = int(time_s * sr / hop)
    column = stft_db[:, t]
    top = np.argsort(column)[-3:][::-1]
    hz_per_bin = sr / frame_length
    freqs = top * hz_per_bin
    print(f"Top-3 loudest frequencies at {time_s}s (the step-3 climax):")
    for i, f in zip(top, freqs):
        print(f"    {f:7.1f} Hz   at {column[i]:6.1f} dB")

def main() -> None:
    demo_intuition()

    y, sr = librosa.load(AUDIO_FILE, sr=SR)
    print("PART 2 - the real song")

    D = librosa.stft(y, n_fft=FRAME, hop_length=HOP)          # complex matrix: freq x time
    db = librosa.amplitude_to_db(np.abs(D), ref=np.max)        # magnitude -> decibels

    hz_per_bin = sr / FRAME
    print(f"  STFT matrix shape (frequency bins x time frames): {db.shape}")
    print(f"  frequency resolution: {hz_per_bin:.1f} Hz per bin (bins go 0..{sr/2:.0f} Hz)")
    print(f"  time resolution: {HOP / sr * 1000:.0f} ms per frame")

    loudest_frequencies_at(db, sr, HOP, FRAME, 56.6)

    plt.figure(figsize=(12, 5))
    img = librosa.display.specshow(
        db, sr=sr, hop_length=HOP, x_axis="time", y_axis="hz", cmap="magma", ax=plt.gca()
    )
    plt.colorbar(img, label="loudness (dB)")
    plt.title("Spectrogram: loudness of each frequency over time")
    plt.tight_layout()
    out = ROOT / "plots" / "05_spectrogram.png"
    plt.savefig(out, dpi=100)
    print(f"\nsaved {out}")

if __name__ == "__main__":
    main()