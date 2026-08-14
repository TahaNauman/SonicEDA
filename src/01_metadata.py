"""Step 1: what does the file itself tell us about the song?"""

from config import AUDIO_FILE
from pathlib import Path
import librosa

def file_stats(path: Path) -> dict:
    """File-level info, no audio decoding needed."""
    return {
        "file_size_bytes": path.stat().st_size,
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
    }

def audio_stats(path: Path) -> dict:
    """Decode the mp3 into raw samples and report the numbers that matter."""
    y, sr = librosa.load(path, mono=True)  # y = samples, sr = samples per second
    duration = librosa.get_duration(y=y, sr=sr)
    return {
        "sample_rate_hz": sr,
        "num_samples": len(y),
        "duration_seconds": round(duration, 2),
        "num_channels": 1,  # mono is the default for librosa.load()
    }

def main() -> None:
    f = file_stats(AUDIO_FILE)
    a = audio_stats(AUDIO_FILE)

    # If we stored every sample as a 32-bit float (4 bytes each):
    raw_size = a["num_samples"] * 4

    print("FILE")
    print(f"  size      : {f['file_size_mb']} MB ({f['file_size_bytes']} bytes)")
    print("\nAUDIO (after decoding)")
    print(f"  sample rate: {a['sample_rate_hz']} samples per second (Hz)")
    print(f"  duration   : {a['duration_seconds']} seconds")
    print(f"  samples    : {a['num_samples']} individual amplitude values")
    print(f"  channels   : {a['num_channels']} (mono)")
    print("\nCOMPRESSION (how much the mp3 shrank the raw data)")
    print(f"  raw size if stored as 32-bit floats: {raw_size / (1024 * 1024):.2f} MB")
    print(f"  mp3 file is {raw_size / f['file_size_bytes']:.0f}x smaller than raw")

if __name__ == "__main__":
    main()