"""Procedural fireplace, loudness mastering (-16 LUFS voice, -3 dB peak) and ducked mix.

The fireplace is synthesized with numpy/scipy when no ./assets/fireplace.* file exists, so the app
is fully self-contained. All noise is generated spectrally / with wrap-around so the loop is seamless.
"""
from __future__ import annotations

import subprocess
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
from scipy import ndimage, signal

import config
from config import AUDIO_SR as SR
from config import log

ASSET_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


# --------------------------------------------------------------------------- loudness (ITU-R BS.1770, gated)
def _k_weight_sos(fs: int) -> np.ndarray:
    G, Q, fc = 3.999843853973347, 0.7071752369554196, 1681.9744509555319
    A = 10 ** (G / 40)
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * Q)
    cw, sA = np.cos(w0), np.sqrt(A)
    b = np.array([A * ((A + 1) + (A - 1) * cw + 2 * sA * alpha),
                  -2 * A * ((A - 1) + (A + 1) * cw),
                  A * ((A + 1) + (A - 1) * cw - 2 * sA * alpha)])
    a = np.array([(A + 1) - (A - 1) * cw + 2 * sA * alpha,
                  2 * ((A - 1) - (A + 1) * cw),
                  (A + 1) - (A - 1) * cw - 2 * sA * alpha])
    shelf = np.concatenate([b / a[0], a / a[0]])
    Q2, fc2 = 0.5003270373238773, 38.13547087613982
    K = np.tan(np.pi * fc2 / fs)
    d = 1 + K / Q2 + K * K
    hp_a = np.array([1.0, 2 * (K * K - 1) / d, (1 - K / Q2 + K * K) / d])
    hp = np.concatenate([[1.0, -2.0, 1.0], hp_a])
    return np.vstack([shelf, hp])


def integrated_lufs(channels: List[np.ndarray], fs: int) -> float:
    sos = _k_weight_sos(fs)
    blk, step = int(0.4 * fs), int(0.1 * fs)
    z_total = None
    for ch in channels:
        y = signal.sosfilt(sos, np.asarray(ch, dtype=np.float64))
        cs = np.concatenate([[0.0], np.cumsum(y * y)])
        idx = np.arange(0, len(y) - blk + 1, step)
        z = (cs[idx + blk] - cs[idx]) / blk
        z_total = z if z_total is None else z_total + z
    if z_total is None or len(z_total) == 0:
        return -70.0
    lb = -0.691 + 10 * np.log10(z_total + 1e-12)
    keep = lb > -70.0
    if not keep.any():
        return -70.0
    rel = -0.691 + 10 * np.log10(z_total[keep].mean()) - 10.0
    keep = lb > max(-70.0, rel)
    return float(-0.691 + 10 * np.log10(z_total[keep].mean())) if keep.any() else -70.0


def soft_limit(x: np.ndarray, ceiling_db: float) -> np.ndarray:
    thr = 10 ** (ceiling_db / 20)
    knee = 0.8 * thr
    ax = np.abs(x)
    over = ax > knee
    if not over.any():
        return x
    y = x.copy()
    y[over] = np.sign(x[over]) * (knee + (thr - knee) * np.tanh((ax[over] - knee) / (thr - knee)))
    return y


# --------------------------------------------------------------------------- fireplace synthesis
def _colored_noise(rng, n: int, exponent: float) -> np.ndarray:
    """FFT noise with power ~ 1/f^exponent (0=white, 1=pink, 2=brown). Circular => loops seamlessly."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / SR)
    f[0] = f[1]
    x = np.fft.irfft(spec / np.power(f, exponent / 2.0), n)
    return x / (x.std() + 1e-12)


def _periodic_filter(sos, x: np.ndarray) -> np.ndarray:
    n = len(x)
    return signal.sosfilt(sos, np.concatenate([x, x, x]))[n:2 * n]


def _periodic_envelope(rng, n: int, rate_hz: float, lo: float, hi: float, smooth_s: float) -> np.ndarray:
    k = max(2, int(n / SR * rate_hz))
    ctrl = rng.uniform(lo, hi, k)
    ctrl = np.append(ctrl, ctrl[0])
    env = np.interp(np.arange(n), np.linspace(0, n, k + 1), ctrl)
    return ndimage.uniform_filter1d(env, size=max(1, int(smooth_s * SR)), mode="wrap")


def _roar(rng, n: int) -> np.ndarray:
    brown = _periodic_filter(signal.butter(4, [28, 260], "bandpass", fs=SR, output="sos"), _colored_noise(rng, n, 2.0))
    brown = brown / brown.std() * _periodic_envelope(rng, n, 0.7, 0.65, 1.35, 0.5)
    pink = _periodic_filter(signal.butter(2, [180, 1800], "bandpass", fs=SR, output="sos"), _colored_noise(rng, n, 1.0))
    pink = pink / pink.std() * 0.22 * _periodic_envelope(rng, n, 1.5, 0.3, 1.4, 0.25)
    hiss = _periodic_filter(signal.butter(2, 3500, "highpass", fs=SR, output="sos"), _colored_noise(rng, n, 0.0))
    hiss = hiss / hiss.std() * 0.03 * _periodic_envelope(rng, n, 3.0, 0.2, 1.5, 0.1)
    return brown + pink + hiss


def _place(buf: np.ndarray, pos: int, burst: np.ndarray, pan: float) -> None:
    idx = (pos + np.arange(len(burst))) % len(buf)
    buf[idx, 0] += burst * np.cos(pan * np.pi / 2)
    buf[idx, 1] += burst * np.sin(pan * np.pi / 2)


def synthesize_fireplace(seconds: float = 60.0, seed: int = 1337) -> np.ndarray:
    """Return a seamless stereo float32 loop (n, 2): warm hearth roar + crackles and pops."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    r1, r2 = _roar(rng, n), _roar(rng, n)
    roar = np.stack([0.75 * r1 + 0.25 * r2, 0.75 * r2 + 0.25 * r1], axis=1)

    activity = _periodic_envelope(rng, n, 0.5, 0.3, 1.7, 1.5)
    amax = activity.max()
    ticks = np.zeros((n, 2))
    for pos in rng.integers(0, n, int(14 * seconds)):
        if rng.random() > activity[pos] / amax:
            continue
        m = int(rng.uniform(0.0008, 0.004) * SR)
        burst = rng.standard_normal(m) * np.exp(-np.arange(m) / (m / rng.uniform(2.5, 5.0)))
        _place(ticks, int(pos), burst * min(0.7, rng.exponential(0.12) + 0.02), rng.uniform(0.15, 0.85))
    ticks = np.stack([_periodic_filter(signal.butter(2, [1200, 8000], "bandpass", fs=SR, output="sos"), ticks[:, c])
                      for c in range(2)], axis=1)

    pops = np.zeros((n, 2))
    for pos in rng.integers(0, n, int(1.0 * seconds)):
        if rng.random() > activity[pos] / amax:
            continue
        m = int(rng.uniform(0.006, 0.03) * SR)
        fc = rng.uniform(700, 3500)
        sos = signal.butter(2, [fc * 0.7, min(fc * 1.4, SR / 2 * 0.9)], "bandpass", fs=SR, output="sos")
        burst = signal.sosfilt(sos, rng.standard_normal(m) * np.exp(-np.arange(m) / (m / rng.uniform(2.0, 4.0))))
        burst = burst / (np.abs(burst).max() + 1e-9) * rng.uniform(0.35, 0.9)
        pan = rng.uniform(0.15, 0.85)
        _place(pops, int(pos), burst, pan)
        if rng.random() < 0.3:  # sizzle cluster after a pop
            for _ in range(rng.integers(4, 10)):
                mm = int(rng.uniform(0.0006, 0.002) * SR)
                b = rng.standard_normal(mm) * np.exp(-np.arange(mm) / (mm / 3.0)) * rng.uniform(0.05, 0.2)
                _place(pops, int(pos + rng.uniform(0.01, 0.07) * SR), signal.sosfilt(sos, b), pan)

    fire = roar * 0.5 + ticks * 2.0 + pops * 2.5
    warm = signal.butter(2, 9000, "lowpass", fs=SR, output="sos")
    fire = np.stack([_periodic_filter(warm, fire[:, c]) for c in range(2)], axis=1)
    return (fire / np.abs(fire).max() * 0.9).astype(np.float32)


# --------------------------------------------------------------------------- external fireplace asset
def find_fireplace_asset() -> Optional[Path]:
    for base in (Path.cwd() / "assets", Path(__file__).resolve().parent / "assets"):
        for ext in ASSET_EXTS:
            p = base / f"fireplace{ext}"
            if p.is_file():
                return p
    return None


def _load_asset(path: Path) -> np.ndarray:
    cmd = [config.find_ffmpeg(), "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).copy()


def loop_to_length(x: np.ndarray, n: int, xfade_s: float) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    L = len(x)
    xs = min(int(xfade_s * SR), L // 4)
    out = np.zeros((n + L, 2), dtype=np.float32)
    out[:L] = x
    pos = L
    if xs > 0:
        t = np.linspace(0, np.pi / 2, xs, dtype=np.float32)[:, None]
        fin, fout = np.sin(t), np.cos(t)
    while pos < n:
        if xs > 0:
            start = pos - xs
            out[start:pos] = out[start:pos] * fout + x[:xs] * fin
            out[pos:start + L] = x[xs:]
            pos = start + L
        else:
            out[pos:pos + L] = x
            pos += L
    return out[:n]


# --------------------------------------------------------------------------- master
def build_master(cfg: config.Settings, timeline: Dict, cache: Path) -> Path:
    out = cache / "master.wav"
    n = int(round(timeline["total_duration"] * SR))

    # --- voice: loudness-normalize to -16 LUFS (dual-mono => +3.01 dB), peak ceiling -3 dB
    voice, vsr = sf.read(str(cache / "voiceover.wav"), dtype="float32")
    voice = voice if voice.ndim == 1 else voice.mean(axis=1)
    lufs_in = integrated_lufs([voice], vsr) + 3.0103
    voice = voice * 10 ** ((config.VOICE_LUFS - lufs_in) / 20)
    g = gcd(SR, vsr)
    voice = signal.resample_poly(voice, SR // g, vsr // g).astype(np.float32)
    voice = soft_limit(voice, config.VOICE_PEAK_DB)
    voice = np.pad(voice, (0, max(0, n - len(voice))))[:n]
    log(f"Voice: {lufs_in:.1f} LUFS -> {config.VOICE_LUFS:.0f} LUFS, peak {20 * np.log10(np.abs(voice).max() + 1e-9):.1f} dBFS")

    # --- fireplace: external asset if present, else procedural
    asset = find_fireplace_asset()
    if asset:
        log(f"Using fireplace asset {asset}")
        loop, xf = _load_asset(asset), 3.0
    else:
        log("No assets/fireplace.* found - synthesizing a procedural fireplace")
        loop, xf = synthesize_fireplace(), 0.0
    fire_lufs = integrated_lufs([loop[:SR * 120, 0], loop[:SR * 120, 1]], SR)
    fire = loop_to_length(loop, n, xf)
    del loop
    fire *= np.float32(10 ** ((config.VOICE_LUFS - cfg.fire_db - fire_lufs) / 20))

    # --- duck: a few dB lower while the narrator speaks, smoothly
    active = ndimage.uniform_filter1d(np.abs(voice), size=int(0.3 * SR)) > 10 ** (-48 / 20)
    mask = ndimage.uniform_filter1d(active.astype(np.float32), size=int(1.0 * SR))
    fire *= (10 ** (-config.FIRE_ACTIVE_EXTRA_DB * mask / 20)).astype(np.float32)[:, None]

    # --- fade fireplace in (3 s) and out (last 5 s)
    fi, fo = min(int(3 * SR), n), min(int(5 * SR), n)
    fire[:fi] *= np.sin(np.linspace(0, np.pi / 2, fi, dtype=np.float32))[:, None]
    fire[n - fo:] *= np.cos(np.linspace(0, np.pi / 2, fo, dtype=np.float32))[:, None]

    mix = fire
    mix += voice[:, None]
    peak = float(np.abs(mix).max())
    if peak > 0.98:
        mix *= 0.98 / peak
    sf.write(str(out), mix, SR, subtype="PCM_16")
    log(f"Master written: {n / SR:.1f}s, fireplace {config.VOICE_LUFS - cfg.fire_db:.0f} LUFS idle / "
        f"{config.VOICE_LUFS - cfg.fire_db - config.FIRE_ACTIVE_EXTRA_DB:.0f} LUFS under speech")
    return out
