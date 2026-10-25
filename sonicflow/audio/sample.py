"""
Sampleo de audio: carga con soundfile (+ fallback WAV stdlib), re-muestreo
de alta calidad con polyphase FFT (scipy) y formas de onda precalculadas.

Todo el costoso trabajo (re-muestreo al sr del motor, waveform) ocurre FUERA
del thread de audio, al importar/cargar una muestra.
"""
from __future__ import annotations

import logging
import os
import wave

import numpy as np

log = logging.getLogger("sonicflow.audio.sample")

try:  # soundfile es la ruta principal (FLAC/OGG/WAV/mp3 vía libsndfile 1.1+)
    import soundfile as sf
    _HAVE_SF = True
except Exception:  # pragma: no cover
    _HAVE_SF = False

try:
    from scipy.signal import resample_poly, upfirdn2
    from scipy.fft import rfft, irfft
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".caf", ".w64"}


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Fallback sin scipy: interpolación lineal vectorizada (aceptable en calidad)."""
    n_in = x.shape[0]
    n_out = max(1, int(round(n_in * sr_out / sr_in)))
    idx = np.linspace(0, n_in - 1, n_out)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_in - 1)
    frac = (idx - i0).astype(np.float32)[:, None] if x.ndim > 1 else (idx - i0).astype(np.float32)
    return (x[i0] * (1 - frac) + x[i1] * frac).astype(np.float32)


def resample_high_quality(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Re-muestreo polyphase con filtro antialias Kaiser-beta=8.5 (~140 dB).

    Para ratios 'feasibles' usa resample_poly; para ratios extremos o señales
    cortas aplica un método FFT-equivalente mediante upfirdn con ventana.
    """
    if sr_in == sr_out or x.shape[0] == 0:
        return x.astype(np.float32, copy=True)
    if not _HAVE_SCIPY:
        return resample_linear(x, sr_in, sr_out)
    g = _gcd(int(sr_in), int(sr_out))
    up, down = int(sr_out // g), int(sr_in // g)
    # limita factores enormes (p.ej. 44100->192000 => up/down grandes pero ok)
    if up > 512 or down > 512:
        # ratio extremo: dos etapas
        mid = sr_out if up <= 512 and down <= 512 else 96000
        y = resample_poly(x.astype(np.float64), int(mid // _gcd(mid, sr_in)),
                          int(sr_in // _gcd(mid, sr_in)))
        y = resample_poly(y, int(sr_out // _gcd(sr_out, int(mid))),
                          int(mid // _gcd(sr_out, int(mid))))
        return y.astype(np.float32)
    half_len = 10 * max(up, down)
    beta = 8.5
    taps = np.sinc(np.arange(-half_len, half_len + 1) / max(up, down)) \
        * np.kaiser(2 * half_len + 1, beta) / max(up, down)
    if x.ndim == 1:
        return upfirdn2(taps, x.astype(np.float64), up=up, down=down)[
            half_len // down: half_len // down + int(round(x.shape[0] * sr_out / sr_in))
        ].astype(np.float32)
    out_len = int(round(x.shape[0] * sr_out / sr_in))
    cols = []
    for ch in range(x.shape[1]):
        y = upfirdn2(taps, x[:, ch].astype(np.float64), up=up, down=down)
        cols.append(y[half_len // down: half_len // down + out_len])
    return np.stack(cols, axis=1).astype(np.float32)


def load_audio_file(path: str) -> tuple[np.ndarray, int]:
    """Carga cualquier archivo soportado -> (float32 [N, C], sr)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")
    ext = os.path.splitext(path)[1].lower()
    if _HAVE_SF:
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            return np.ascontiguousarray(data), int(sr)
        except Exception as exc:
            log.warning("soundfile no pudo leer %s (%s); intentando wave/ffmpeg", path, exc)
    if ext == ".wav":
        try:
            with wave.open(path, "rb") as wf:
                sr = wf.getframerate()
                nch = wf.getnchannels()
                sw = wf.getsampwidth()
                raw = wf.readframes(wf.getnframes())
            dt = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.int32}[sw]
            a = np.frombuffer(raw, dtype=dt)
            if sw == 3:  # int24
                a = (a << 8).astype(np.int32)
                a = np.right_shift(a.view(np.int32), 8) if False else a
            if sw == 1:
                a = a.astype(np.float32) - 128.0
                a /= 127.0
            else:
                a = a.astype(np.float32) / float(2 ** (8 * sw - 1))
            a = a.reshape(-1, nch)
            return np.ascontiguousarray(a), sr
        except Exception as exc:
            log.warning("wave stdlib falló en %s: %s", path, exc)
    # último recurso: ffmpeg -> wav temporal (formatos que libsndfile no soporta)
    return _ffmpeg_decode(path)


def _ffmpeg_decode(path: str) -> tuple[np.ndarray, int]:
    import shutil
    import subprocess
    import tempfile
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError(
            f"Formato no soportado nativamente: {os.path.basename(path)}. "
            "Instala ffmpeg o convierte el archivo a WAV.")
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", path,
                        "-c:a", "pcm_f32le", tmp], check=True, timeout=120)
        data, sr = sf.read(tmp, dtype="float32", always_2d=True) if _HAVE_SF else (None, 0)
        if data is None:
            with wave.open(tmp, "rb") as wf:
                sr = wf.getframerate()
                raw = wf.readframes(wf.getnframes())
            data = np.frombuffer(raw, dtype=np.float32).reshape(-1, wf.getnchannels())
        return np.ascontiguousarray(data), int(sr)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def compute_waveform(data: np.ndarray, buckets: int = 2000) -> np.ndarray:
    """(buckets, 2) min/max por bucket para dibujar la forma de onda."""
    n = data.shape[0]
    if n == 0:
        return np.zeros((buckets, 2), dtype=np.float32)
    size = max(1, n // buckets)
    usable = (n // size) * size
    reshaped = data[:usable].reshape(-1, size, data.shape[1] if data.ndim > 1 else 1)
    mn = reshaped.min(axis=1)
    mx = reshaped.max(axis=1)
    out = np.stack([mn[:, 0], mx[:, 0]], axis=1)
    if out.shape[0] < buckets:  # padding
        pad = np.zeros((buckets - out.shape[0], 2), dtype=np.float32)
        out = np.concatenate([out, pad], axis=0)
    return out.astype(np.float32)[:buckets]


class Sample:
    """Una muestra cargada en memoria, lista para reproducir en el motor.

    `data`: float32 [N, 2] (estéreo, canales duplicados si era mono).
    `sr`  : sample rate NATIVO (la playback rate relativa se calcula contra el
            sr del motor en el Voice).
    `peak`: pico máximo para normalización/ganancia sana.
    """

    __slots__ = ("name", "path", "data", "sr", "duration", "peak", "waveform", "key_cent")

    def __init__(self, name: str, data: np.ndarray, sr: int, path: str = "") -> None:
        if data.ndim == 1:
            data = data[:, None]
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        elif data.shape[1] > 2:
            data = data[:, :2]
        self.name = name
        self.path = path
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.sr = int(sr)
        self.duration = self.data.shape[0] / float(sr)
        self.peak = float(np.max(np.abs(self.data))) if self.data.size else 0.0
        self.waveform = compute_waveform(self.data)
        self.key_cent = 0  # nota raíz en cents respecto a C (para afinación)

    @classmethod
    def from_file(cls, path: str, target_sr: int | None = None) -> "Sample":
        data, sr = load_audio_file(path)
        if data.size == 0:
            raise ValueError(f"Archivo de audio vacío: {path}")
        name = os.path.splitext(os.path.basename(path))[0]
        s = cls(name, data, sr, path)
        return s

    def resampled(self, target_sr: int) -> "Sample":
        """Devuelve una copia remuestreada (calidad alta) al sr del motor."""
        if target_sr == self.sr:
            return self
        new = Sample(self.name, resample_high_quality(self.data, self.sr, target_sr),
                     target_sr, self.path)
        new.key_cent = self.key_cent
        return new

    def normalized_copy(self, peak_target: float = 0.9) -> "Sample":
        if self.peak <= 0:
            return self
        g = peak_target / self.peak
        return Sample(self.name, (self.data * g).astype(np.float32), self.sr, self.path)
