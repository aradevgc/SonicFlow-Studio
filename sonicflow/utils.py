"""Utilidades comunes: timecode, escritura atómica de JSON, logging."""
from __future__ import annotations

import json
import logging
import os
import tempfile

log = logging.getLogger("sonicflow")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# --------------------------------------------------------------------------- #
#  Timecode
# --------------------------------------------------------------------------- #
def seconds_to_timecode(seconds: float) -> str:
    """'HH:MM:SS.mmm'"""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def seconds_to_smpte(seconds: float, fps: float) -> str:
    """Timecode SMPTE no-drop HH:MM:SS:FF para una fps dada."""
    if fps <= 0:
        fps = 25.0
    total_frames = int(round(seconds * fps))
    fpint = int(round(fps))
    f = total_frames % fpint
    t = total_frames // fpint
    s = t % 60
    m = (t // 60) % 60
    h = t // 3600
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def parse_timecode(text: str) -> float | None:
    """Admite 'SS', 'MM:SS', 'HH:MM:SS', 'HH:MM:SS.mmm' y SMPTE 'HH:MM:SS:FF'."""
    text = text.strip()
    if not text:
        return None
    try:
        parts = text.replace(".", ":").split(":")
        secs = 0.0
        mult = 1.0
        for p in reversed(parts):
            secs += float(p) * mult
            mult *= 60.0
        # 'HH:MM:SS:mmm' con 4 campos separados por '.'->mmm era milisegundos
        if "." in text and len(text.split(":")) == 3:
            pass  # ya tratado: último campo 'SS.mmm' como float
        elif len(parts) == 4 and "." not in text:
            # SMPTE HH:MM:SS:FF asumido 25 fps si no se indica; devolvemos base
            secs = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            secs += float(parts[3]) / 25.0
        return secs
    except Exception:
        return None


# --------------------------------------------------------------------------- #
#  JSON atómico
# --------------------------------------------------------------------------- #
def atomic_write_json(path: str, data: dict, indent: int = 2) -> None:
    """Escribe JSON de forma atómica (archivo temporal + rename)."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=indent, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def read_json(path: str, default=None):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as exc:
        log.warning("JSON corrupto en %s: %s", path, exc)
        return default


# --------------------------------------------------------------------------- #
#  Varios
# --------------------------------------------------------------------------- #
def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def note_name(midi_note: float) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    n = int(round(midi_note))
    return f"{names[n % 12]}{n // 12 - 1}"
