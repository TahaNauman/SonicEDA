# SonicEDA

Sonic Exploratory Data Analysis — a learning experiment to discover what
data can be extracted from an .mp3 song file using digital signal
processing, audio feature extraction, and data visualization.

Every step is a plain `.py` script (no notebooks) that extracts one type of
data from an audio file.

## Project layout

```
SonicEDA/
├── data/        # raw audio files (gitignored)
├── plots/       # generated visualizations (gitignored)
├── tests/       # tests
└── 0X_*.py      # one script per concept, run directly
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Learning steps

| # | Script | Data extracted |
|---|--------|----------------|
| 1 | `01_metadata.py` | duration, sample rate, bit depth, channels; compression ratio |
| 2 | `02_waveform.py` | raw samples → amplitude over time |
| 3 | `03_rms_energy.py` | loudness envelope |
| 4 | `04_zero_crossings.py` | brightness / noise estimate |
| 5 | `05_spectrogram.py` | frequencies over time (STFT) |
| 6 | `06_features.py` | tempo, beats, spectral stats, MFCCs, chroma |
| 7 | `07_eda.py` | all features in a pandas DataFrame |
| 8 | `08_visualize.py` | plots of each feature |
| 9 | `09_summary.py` | plain-language menu of all features |
