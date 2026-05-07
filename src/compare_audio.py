#!/usr/bin/env python3
"""
compare_audio.py

Compare acoustic properties of two wav files to diagnose
why a speaker embedding might not be matching.

Usage:
    python src/compare_audio.py \
        --clip1 data/hosts/host_clip.wav \
        --clip2 data/raw_audio/ep_098.wav
"""

import argparse
import wave
import struct
import math
from pathlib import Path

import numpy as np
from scipy import signal


def load_wav_mono(path: Path, max_seconds: float = 120.0):
    """Load a wav file as mono float32, up to max_seconds."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        duration = n_frames / sample_rate

        # Limit to max_seconds
        frames_to_read = min(n_frames, int(max_seconds * sample_rate))
        raw = wf.readframes(frames_to_read)

    # Unpack to int16
    fmt = f"{len(raw) // sample_width}h"
    samples = struct.unpack(fmt, raw)
    audio = np.array(samples, dtype=np.float32) / 32768.0

    # Mix down to mono if stereo
    if n_channels == 2:
        audio = audio[0::2] * 0.5 + audio[1::2] * 0.5

    return audio, sample_rate, duration, n_channels


def rms_db(audio: np.ndarray) -> float:
    rms = np.sqrt(np.mean(audio ** 2))
    return 20 * np.log10(rms + 1e-10)


def peak_db(audio: np.ndarray) -> float:
    return 20 * np.log10(np.max(np.abs(audio)) + 1e-10)


def silence_ratio(audio: np.ndarray, threshold_db: float = -40.0) -> float:
    """Fraction of 10ms frames below threshold."""
    frame_size = 160  # 10ms at 16kHz
    silent = 0
    total = 0
    for i in range(0, len(audio) - frame_size, frame_size):
        frame = audio[i:i + frame_size]
        db = 20 * np.log10(np.sqrt(np.mean(frame ** 2)) + 1e-10)
        if db < threshold_db:
            silent += 1
        total += 1
    return silent / total if total > 0 else 0.0


def spectral_centroid(audio: np.ndarray, sr: int) -> float:
    """Average frequency weighted by energy."""
    freqs, psd = signal.welch(audio, sr, nperseg=1024)
    if psd.sum() == 0:
        return 0.0
    return float(np.sum(freqs * psd) / np.sum(psd))


def spectral_rolloff(audio: np.ndarray, sr: int, pct: float = 0.85) -> float:
    """Frequency below which pct% of energy is contained."""
    freqs, psd = signal.welch(audio, sr, nperseg=1024)
    cumsum = np.cumsum(psd)
    threshold = pct * cumsum[-1]
    idx = np.searchsorted(cumsum, threshold)
    return float(freqs[min(idx, len(freqs) - 1)])


def snr_estimate(audio: np.ndarray) -> float:
    """
    Rough SNR estimate: ratio of voiced frame energy to silent frame energy.
    """
    frame_size = 1600  # 100ms at 16kHz
    voiced, silent = [], []
    for i in range(0, len(audio) - frame_size, frame_size):
        frame = audio[i:i + frame_size]
        e = np.mean(frame ** 2)
        db = 10 * np.log10(e + 1e-10)
        if db > -40:
            voiced.append(e)
        else:
            silent.append(e)

    if not voiced or not silent:
        return float("inf")

    return 10 * np.log10(np.mean(voiced) / (np.mean(silent) + 1e-10))


def estimate_pitch(audio: np.ndarray, sr: int) -> tuple[float, float]:
    """
    Estimate median and std of fundamental frequency using autocorrelation.
    Returns (median_f0_hz, std_f0_hz).
    """
    # Resample to 16kHz for pitch analysis
    if sr != 16000:
        audio = signal.resample_poly(audio, 16000, sr)
        sr = 16000

    frame_len = int(0.03 * sr)
    hop = int(0.01 * sr)
    min_lag = int(sr / 400)
    max_lag = int(sr / 60)

    pitches = []
    for i in range(0, len(audio) - frame_len, hop):
        frame = audio[i:i + frame_len]
        if np.sqrt(np.mean(frame ** 2)) < 0.01:
            continue
        corr = np.correlate(frame, frame, "full")
        corr = corr[len(corr) // 2:]
        if max_lag >= len(corr):
            continue
        sub = corr[min_lag:max_lag]
        if len(sub) == 0:
            continue
        peak_idx = np.argmax(sub) + min_lag
        if corr[0] > 0 and corr[peak_idx] / corr[0] > 0.3:
            pitches.append(sr / peak_idx)

    if not pitches:
        return 0.0, 0.0
    return float(np.median(pitches)), float(np.std(pitches))


def band_energy_ratio(audio: np.ndarray, sr: int) -> dict:
    """Energy ratios across frequency bands."""
    freqs, psd = signal.welch(audio, sr, nperseg=2048)
    total = psd.sum() + 1e-10

    bands = {
        "sub_bass (0-80Hz)":     (0, 80),
        "speech (80-3400Hz)":    (80, 3400),
        "presence (3.4-8kHz)":   (3400, 8000),
        "air (8kHz+)":           (8000, sr // 2),
    }

    result = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        result[name] = float(psd[mask].sum() / total)
    return result


def compare(path1: Path, path2: Path):
    print(f"\nLoading clips...")
    audio1, sr1, dur1, ch1 = load_wav_mono(path1)
    audio2, sr2, dur2, ch2 = load_wav_mono(path2)

    # Resample clip2 to match clip1 sr if different
    if sr1 != sr2:
        audio2 = signal.resample_poly(audio2, sr1, sr2)
        sr2 = sr1

    print(f"\n{'='*60}")
    print(f"{'Property':<35} {'Clip 1':>10} {'Clip 2':>10} {'Match?':>8}")
    print(f"{'='*60}")

    def row(label, v1, v2, fmt=".1f", tolerance=None):
        v1_str = f"{v1:{fmt}}"
        v2_str = f"{v2:{fmt}}"
        if tolerance is not None:
            match = "✓" if abs(v1 - v2) <= tolerance else "✗"
        else:
            match = ""
        print(f"{label:<35} {v1_str:>10} {v2_str:>10} {match:>8}")

    # Basic properties
    print(f"\n--- Basic Properties ---")
    row("Duration (s)",          dur1,  dur2,  fmt=".1f")
    row("Sample rate (Hz)",      sr1,   sr2,   fmt=".0f", tolerance=0)
    row("Channels",              ch1,   ch2,   fmt=".0f", tolerance=0)

    # Loudness
    print(f"\n--- Loudness ---")
    rms1, rms2 = rms_db(audio1), rms_db(audio2)
    peak1, peak2 = peak_db(audio1), peak_db(audio2)
    row("RMS level (dB)",        rms1,  rms2,  fmt=".1f", tolerance=6.0)
    row("Peak level (dB)",       peak1, peak2, fmt=".1f", tolerance=6.0)

    # Silence
    print(f"\n--- Silence / Activity ---")
    sil1, sil2 = silence_ratio(audio1), silence_ratio(audio2)
    snr1, snr2 = snr_estimate(audio1), snr_estimate(audio2)
    row("Silence ratio",         sil1,  sil2,  fmt=".2f", tolerance=0.2)
    row("SNR estimate (dB)",     snr1,  snr2,  fmt=".1f", tolerance=10.0)

    # Pitch
    print(f"\n--- Pitch (F0) ---")
    p1, ps1 = estimate_pitch(audio1, sr1)
    p2, ps2 = estimate_pitch(audio2, sr2)
    row("Median F0 (Hz)",        p1,    p2,    fmt=".1f", tolerance=30.0)
    row("F0 std dev (Hz)",       ps1,   ps2,   fmt=".1f", tolerance=20.0)

    # Spectral
    print(f"\n--- Spectral Characteristics ---")
    sc1 = spectral_centroid(audio1, sr1)
    sc2 = spectral_centroid(audio2, sr2)
    sr85_1 = spectral_rolloff(audio1, sr1)
    sr85_2 = spectral_rolloff(audio2, sr2)
    row("Spectral centroid (Hz)", sc1,  sc2,   fmt=".0f", tolerance=300.0)
    row("Spectral rolloff 85% (Hz)", sr85_1, sr85_2, fmt=".0f", tolerance=1000.0)

    # Band energy
    print(f"\n--- Frequency Band Energy Ratios ---")
    be1 = band_energy_ratio(audio1, sr1)
    be2 = band_energy_ratio(audio2, sr2)
    for band in be1:
        row(band, be1[band], be2[band], fmt=".3f", tolerance=0.1)

    print(f"\n{'='*60}")
    print(f"Files:")
    print(f"  Clip 1: {path1}")
    print(f"  Clip 2: {path2}")
    print(f"\n✓ = within tolerance  ✗ = significant difference")
    print(f"\nKey things to look for:")
    print(f"  - RMS difference > 6dB  → loudness mismatch")
    print(f"  - F0 difference > 30Hz  → possibly different speakers OR vocal change")
    print(f"  - Band energy mismatch  → different microphones or codecs")
    print(f"  - Silence ratio > 0.5   → too much silence in clip, poor enrollment sample")
    print(f"  - SNR < 10dB            → noisy clip, unreliable embedding")


def main():
    ap = argparse.ArgumentParser(
        description="Compare acoustic properties of two wav files"
    )
    ap.add_argument("--clip1", required=True, help="First wav file (e.g. enrollment clip)")
    ap.add_argument("--clip2", required=True, help="Second wav file (e.g. podcast episode)")
    args = ap.parse_args()

    path1 = Path(args.clip1)
    path2 = Path(args.clip2)

    for p in [path1, path2]:
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    compare(path1, path2)


if __name__ == "__main__":
    main()
