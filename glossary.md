# Glossary

Every concept used across the project, written in plain language.
One section per script, growing as we go.

## 00_common_ideas.py

Concepts that are used in (almost) every script.

- **MP3 (audio file)** — a song saved on disk in a *compressed* format. The computer can't play a file directly; it first *decodes* it into raw numbers.
- **Decoding (loading)** — turning the compressed file back into a long list of amplitude numbers. `librosa.load()` does this.
- **librosa** — a Python library for audio/music analysis. It loads audio and computes all the features in this project.
- **numpy array** — Python's high-performance list of numbers. Audio samples live in one of these.

## 01_metadata.py

What the file itself tells us about the song, before any math.

- **Amplitude** — the loudness of the air pressure at one instant. Each sample value (`y[i]`) is one amplitude, in the range roughly −1 to +1.
- **Sample** — one single amplitude measurement. The song is 4,349,376 samples.
- **Sample rate (`sr`)** — how many amplitude measurements we take per second (22,050). Sets time resolution: the more samples per second, the finer we slice time — and the higher the pitch we can represent.
- **Duration** — total length in seconds: `num_samples / sample_rate` (4,349,376 / 22,050 = 197.25 s).
- **Mono / channels** — librosa always combines left+right into a single channel (mono) for simplicity.
- **Compression / lossy** — MP3 throws away data our ears barely hear, so the *file on disk* (2.39 MB) is much smaller than the *decoded raw samples* (16.6 MB if stored as 32-bit floats). That's the 7× ratio. The decoded sound is an *approximation* of the original, not identical.
- **`y` and `sr`** — the two things `librosa.load(path)` returns: `y` = the array of sample amplitudes; `sr` = the sample rate that tells us how to interpret those numbers (each index = 1/22050 of a second).

## 02_waveform.py

What the raw sound *looks* like.

- **Waveform** — a graph of amplitude over time. x-axis = seconds, y-axis = amplitude (−1 to +1). The shape of the line *is* the sound, drawn instead of played.
- **Peaks** — the loudest single samples (≈ +1.01 / −1.05 here). Digital audio is bounded near ±1.
- **Silence** — near-zero amplitude (the first 15 samples were all 0.0000 — the song starts silent). Shows up as a flat line at 0 on the plot.
- **`librosa.times_like(y, sr=sr)`** — builds an array of *seconds* values (one per sample) so we can plot time on the x-axis instead of raw sample indices.
- **RMS** — Root Mean Square: the average loudness. 0.256 here means the song is "medium loud" overall. (How this is computed *per moment* is step 3.)
- **Reading the plot** — flat line = quiet, tall squiggles = loud. The loud/quiet sections of the song are visible as the shape's thickness.

## 03_rms_energy.py

Loudness as a curve over time, not one flat average.

- **RMS (Root Mean Square)** — one number summarizing a frame's loudness: `sqrt(mean(frame²))`. Square each sample (kills negatives so louder samples dominate), average, then square-root back to amplitude units. 0.256 overall; up to 0.434 at the loudest moment.
- **Framing (windowing)** — chopping the 197 s of audio into short overlapping chunks called frames, so we get one loudness value *per moment* instead of for the whole song. Without frames, we'd only have a single average.
- **frame_length** — samples per window (2,048 ≈ 93 ms at 22,050 Hz).
- **hop_length** — how many samples each window steps forward (1,024 ≈ 46 ms). Hop < frame = **50% overlap**, which smooths the curve (adjacent frames share half their samples).
- **Envelope** — the resulting per-time curve of RMS values (4,246 frames → one per ~46 ms). "How loud, moment by moment."
- **frames_to_time** — converts a frame index into seconds (`index × hop / sr`) so we can plot the envelope on the time axis.
- **center (zero-padding)** — librosa's `center=True` pads the signal (by frame_length/2) so frames align on the timeline; we used `center=False` so frames start at sample 0. The 2-frame count difference and the shifted comparison (0.25 diff) were both symptoms of this.
- **sliding_window_view** — numpy's way of cutting many overlapping windows at once (one row per window), the vectorized trick behind the hand-written `rms_by_hand`.

## 04_zero_crossings.py

How often the signal flips sign — a cheap brightness/timbre measurement.

- **Zero crossing** — the signal switching from positive to negative amplitude (or vice versa). Count them to gauge how fast the signal oscillates.
- **Brightness proxy** — high-frequency/fizzy/noisy sounds cross zero often; low/warm/tonal sounds cross rarely. ~2,291 sign flips per second here = a busy, moderately bright signal.
- **`(y[:-1] > 0) != (y[1:] > 0)`** — the one-line trick: compare each sample's sign to its neighbor's; `!=` is True exactly where the sign flips.
- **Rate vs count** — `flips.mean()` divides by the number of samples, so the number is comparable across different-length audio (crossings *per sample*). `rate × sr` converts to crossings *per second*.
- **threshold / pad (librosa's knobs)** — librosa's `zero_crossings` by default ignores flips that stay under a tiny magnitude (`threshold=1e-10`) and pads frame edges (`pad=True`). Setting `threshold=0` matched our hand version to within 2.5e-3; the residual is frame-edge handling.
- **RMS vs ZCR independence** — loudness (RMS) and brightness (ZCR) measure *different* things: a loud tonal hum can have low ZCR, a quiet hiss high ZCR. The scatter plot of the two per-frame values is a blob for exactly that reason.

## 05_spectrogram.py

At each moment, how loud is each frequency? The "money picture" of the project.

- **Frequency (Hz)** — how many oscillations per second; the pitch of a note. 440 Hz = musical A4.
- **Sine wave / pure tone** — a wave of a single frequency; the "atom" of sound. `np.sin(2 * np.pi * freq * t)`.
- **Superposition** — music is many pure tones *added together*. 440 + 880 Hz sums into a jagged wave that is neither but contains both.
- **Correlation (the heart of Fourier)** — slide a candidate tone across the signal and measure how well it fits (dot product). High match (0.50) = that tone is in the sound; near-zero (0.00) = it isn't. Scanning this across all frequencies IS the Fourier transform.
- **Fourier transform** — decomposes a chunk of sound into "which frequencies, and how loud each one." Doesn't add or remove information; just re-expresses it in terms of pure tones.
- **STFT (Short-Time Fourier Transform)** — the Fourier transform applied to overlapping frames (same framing as steps 3–4), producing a matrix of frequency × time. `librosa.stft(y, n_fft, hop_length)`.
- **STFT matrix** — here (1,025 freq bins × 4,248 time frames): 1,025 frequencies measured every 46 ms.
- **Frequency bin / resolution** — how finely the frequency axis is sliced; `sr / n_fft` = 10.8 Hz per bin here. Longer frames = finer frequency, coarser time (the fundamental resolution tradeoff).
- **dB (decibel) / log scale** — `amplitude_to_db` compresses the huge range of loudness so quiet AND loud details are both visible on the heatmap.
- **Spectrogram** — the resulting picture: x = time, y = frequency, color = loudness. Reading it: bright bottom bands = bass, upper regions = harmonic detail; a bright vertical line = a loud moment.
- **Reality anchor** — at 56.6 s (step 3's climax) the song is mostly 86 Hz (deep bass) + 850/904 Hz (bright body): "loud" = a stack of specific frequencies, not one number.

## 06_features.py

Distilling the song into a small set of meaningful numbers — the "fingerprint."

- **Feature** — a number (or short array) that summarizes raw data. The whole song (4.3M samples) compresses into ~7 headline numbers: tempo, beats, centroid, rolloff, bandwidth, 13 MFCCs, 12 chroma values.
- **Tempo (BPM)** — beats per minute; the speed/danceability. Pokemon.mp3 = 143.6 BPM = fast/upbeat.
- **Beat tracking** — `librosa.beat.beat_track` finds where the pulse lands (466 beats over 197 s). Returns a 2-tuple `(tempo, beat_frames)`; tempo is a 1-element array here, so flatten with `float(np.asarray(tempo).mean())`.
- **Spectral centroid** — the "center of mass" of the spectrum (weighted average frequency). Higher = brighter. 2,362 Hz avg, peaking at ~6.4 kHz.
- **Spectral rolloff** — the frequency below which ~85% of the energy sits. 5,324 Hz here → the bright energy is concentrated, not spread wide.
- **Spectral bandwidth** — how spread out the energy is around the centroid. 2,589 Hz → a fairly centered/tonal sound.
- **MFCC (Mel-frequency cepstral coefficients)** — 13 coefficients capturing *timbre* (vocal/instrument color). Survives compression and identifies "what kind of voice/gear" — the classic sound fingerprint. Coefficient 0 = overall energy; the rest encode spectral shape.
- **Chroma** — energy in each of the 12 pitch classes (C..B). Pokemon.mp3 leans D, F, G → its tonal/harmonic emphasis. Not a proof of key, but a real measure of which notes dominate.
- **`librosa.frames_to_time`** — converts beat/frame indices back to seconds for plotting on the time axis.

## 08_visualize.py

*(not yet built)*

## 09_summary.py

*(not yet built)*