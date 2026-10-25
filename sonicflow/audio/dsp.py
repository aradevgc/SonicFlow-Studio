"""
Primitivas DSP usadas por las capas del motor. Implementadas con buffers de
estado mutable (np arrays) para poder ejecutarse dentro del audio callback
sin asignaciones relevantes por bloque. Todas son deterministas y estables.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
#  Filtro biquad (RBJ Audio EQ Cookbook) — directo II transpuesto
# --------------------------------------------------------------------------- #
class Biquad:
    def __init__(self, fs: float) -> None:
        self.fs = fs
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0
        self.b0 = 1.0
        self.b1 = self.b2 = self.a1 = self.a2 = 0.0

    def set_lowpass(self, freq: float, q: float = 0.707) -> None:
        w0 = 2 * np.pi * min(freq, self.fs * 0.49) / self.fs
        cw, sw = np.cos(w0), np.sin(w0)
        alpha = sw / (2 * max(q, 0.1))
        a0 = 1 + alpha
        self.b0, self.b1, self.b2 = (1 - cw) / 2 / a0, (1 - cw) / a0, (1 - cw) / 2 / a0
        self.a1, self.a2 = 2 * cw / a0 * -1, (1 - alpha) / a0

    def set_highpass(self, freq: float, q: float = 0.707) -> None:
        w0 = 2 * np.pi * min(freq, self.fs * 0.49) / self.fs
        cw, sw = np.cos(w0), np.sin(w0)
        alpha = sw / (2 * max(q, 0.1))
        a0 = 1 + alpha
        self.b0, self.b1, self.b2 = (1 + cw) / 2 / a0, -(1 + cw) / a0, (1 + cw) / 2 / a0
        self.a1, self.a2 = -2 * cw / a0, (1 - alpha) / a0

    def set_bandpass(self, freq: float, q: float = 1.0) -> None:
        w0 = 2 * np.pi * min(max(freq, 10), self.fs * 0.49) / self.fs
        cw, sw = np.cos(w0), np.sin(w0)
        alpha = sw / (2 * max(q, 0.1))
        a0 = 1 + alpha
        self.b0, self.b1, self.b2 = sw / 2 / a0, 0.0, -sw / 2 / a0
        self.a1, self.a2 = -2 * cw / a0, (1 - alpha) / a0

    def reset(self) -> None:
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def process_block(self, x: np.ndarray) -> np.ndarray:
        """Direct-form II transposed, procesado bloque-a-bloque con estado."""
        y = np.empty_like(x)
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        x1, x2, y1, y2 = self.x1, self.x2, self.y1, self.y2
        # loop python por muestra: suficiente para bloques de <=2048 y CPU actual;
        # coste O(N). Se optimiza con numba si hiciera falta.
        for i in range(x.shape[0]):
            xi = x[i]
            yi = b0 * xi + x1
            x1 = b1 * xi - a1 * y1 + x2
            x2 = b2 * xi - a2 * y2
            y2 = y1
            y1 = yi
            y[i] = yi
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        return y


# --------------------------------------------------------------------------- #
#  Envolvente ADSR (por voz, barata)
# --------------------------------------------------------------------------- #
class AdsrState:
    __slots__ = ("stage", "level")

    def __init__(self) -> None:
        self.stage = 0  # 0 attack, 1 decay, 2 sustain, 3 release
        self.level = 0.0


def adsr_process(state: AdsrState, block_n: int, a: float, d: float, s: float,
                 r: float, sr: float, gate: bool, finished: bool) -> np.ndarray:
    """Genera la envolvente para un bloque. gate=False tras finish => release."""
    env = np.empty(block_n, dtype=np.float32)
    inv = 1.0 / max(sr, 1.0)
    pos = 0
    lvl = state.level
    stage = state.stage
    if not gate:
        stage = 3
    while pos < block_n:
        remaining = block_n - pos
        if stage == 0:  # attack
            inc = (1.0 - lvl) / max(a * sr, 1.0)
            steps = min(remaining, int(np.ceil((1.0 - lvl) / max(inc, 1e-9))))
            for k in range(steps):
                lvl = min(1.0, lvl + inc)
                env[pos + k] = lvl
            pos += steps
            if lvl >= 1.0 - 1e-6:
                stage = 1
        if stage == 1 and pos < block_n:  # decay hacia sustain
            inc = max(s - lvl, -1e-9) / max(d * sr, 1.0)
            steps = min(block_n - pos, int(np.ceil(abs((s - lvl) / inc))) if inc != 0 else remaining)
            for k in range(steps):
                lvl += inc
                env[pos + k] = lvl
            pos += steps
            if abs(lvl - s) < 1e-5:
                lvl = s
                stage = 2
        if stage == 2 and pos < block_n:  # sustain
            n = block_n - pos
            env[pos:pos + n] = s
            lvl = s
            pos += n
        if stage == 3 and pos < block_n:  # release
            inc = -lvl / max(r * sr, 1.0)
            steps = min(block_n - pos, int(np.ceil(lvl / max(-inc, 1e-9))))
            for k in range(steps):
                lvl = max(0.0, lvl + inc)
                env[pos + k] = lvl
            pos += steps
            if lvl <= 1e-6:
                lvl = 0.0
                pos = block_n  # terminada
                break
    # relleno por si acaso
    if pos < block_n:
        env[pos:] = lvl
    state.level = lvl
    state.stage = stage
    return env


# --------------------------------------------------------------------------- #
#  Saturación tanh con pre/post gain compensado
# --------------------------------------------------------------------------- #
def saturate_tanh(block: np.ndarray, drive: float) -> np.ndarray:
    if drive <= 1.01:
        return block
    pre = 1.0 / np.tanh(drive)
    return (np.tanh(block * drive) * pre).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Delay con realimentación (estéreo ping-pong opcional)
# --------------------------------------------------------------------------- #
class StereoDelay:
    def __init__(self, fs: int, max_ms: float = 2000.0) -> None:
        self.fs = fs
        self.buf = np.zeros((int(fs * max_ms / 1000.0) + 4, 2), dtype=np.float32)
        self.mask = self.buf.shape[0] - 1  # requiere pow2
        n = 1
        while n < self.buf.shape[0]:
            n <<= 1
        self.buf = np.zeros((n, 2), dtype=np.float32)
        self.mask = n - 1
        self.wpos = 0

    def reset(self) -> None:
        self.buf[:] = 0.0
        self.wpos = 0

    def process(self, input_lr: np.ndarray, time_ms: float, feedback: float,
                 mix: float, cross: float = 0.0) -> np.ndarray:
        """input_lr: [N,2]. Devuelve [N,2] dry+wet. cross: realimentación cruzada L/R."""
        n = input_lr.shape[0]
        delay = int(np.clip(time_ms * self.fs / 1000.0, 1, self.mask))
        fb = min(feedback, 0.95)
        out = np.empty_like(input_lr)
        buf = self.buf
        mask = self.mask
        w = self.wpos
        for i in range(n):
            r = (w - delay) & mask
            dl, dr = buf[r, 0], buf[r, 1]
            wet_l = dl + dr * cross
            wet_r = dr + dl * cross
            il, ir = input_lr[i, 0], input_lr[i, 1]
            nl = il + wet_l * fb
            nr = ir + wet_r * fb
            buf[w, 0] = nl
            buf[w, 1] = nr
            m = mix
            out[i, 0] = il * (1 - m) + wet_l * m
            out[i, 1] = ir * (1 - m) + wet_r * m
            w = (w + 1) & mask
        self.wpos = w
        return out


# --------------------------------------------------------------------------- #
#  Reverberación Schroeder clásica (8 comb + 2 allpass por canal)
# --------------------------------------------------------------------------- #
class Reverb:
    _COMBS_L = (1557, 1617, 1491, 1422)
    _COMBS_R = (1601, 1543, 1627, 1489)
    _ALLPASS_L = (225, 556)
    _ALLPASS_R = (480, 211)

    def __init__(self, fs: int) -> None:
        self.fs = fs
        scale = fs / 44100.0
        self.combs = [np.zeros(int(c * scale) + 2, dtype=np.float32)
                      for c in self._COMBS_L + self._COMBS_R]
        self.allpass = [np.zeros(int(a * scale) + 2, dtype=np.float32)
                        for a in self._ALLPASS_L + self._ALLPASS_R]
        self.cidx = [0] * 8
        self.apidx = [0] * 4
        self.damp = 0.4
        self.hpf = Biquad(fs)
        self.hpf.set_highpass(45.0, 0.5)

    def reset(self) -> None:
        for b in self.combs + self.allpass:
            b[:] = 0.0

    def process(self, xy: np.ndarray, mix: float, room: float) -> np.ndarray:
        """xy [N,2] -> [N,2] con reverberación mono-in estéreo-out."""
        n = xy.shape[0]
        mono = ((xy[:, 0] + xy[:, 1]) * 0.5)
        dry = xy.copy()
        wet_in = self.hpf.process_block(mono) * 0.15
        damp = 0.2 + room * 0.7
        fb = 0.7 + room * 0.25
        out_l = np.zeros(n, dtype=np.float32)
        out_r = np.zeros(n, dtype=np.float32)
        combs = self.combs
        cidx = self.cidx
        # 4 comb izquierda, 4 derecha (paralelo)
        acc_l = np.zeros(n, dtype=np.float32)
        acc_r = np.zeros(n, dtype=np.float32)
        for ci in range(4):
            buf = combs[ci]
            ln = len(buf) - 1
            idx = cidx[ci]
            for i in range(n):
                val = buf[idx]
                acc_l[i] += val
                buf[idx] = wet_in[i] + val * fb * (1 - self.damp) 
                # damping one-pole sobre el propio comb
                idx += 1
                if idx >= ln:
                    idx = 0
            cidx[ci] = idx
        for ci in range(4, 8):
            buf = combs[ci]
            ln = len(buf) - 1
            idx = cidx[ci]
            for i in range(n):
                val = buf[idx]
                acc_r[i] += val
                buf[idx] = wet_in[i] + val * fb
                idx += 1
                if idx >= ln:
                    idx = 0
            cidx[ci] = idx
        # allpass en serie por canal
        for ai in range(2):
            buf = self.allpass[ai]
            ln = len(buf) - 1
            idx = self.apidx[ai]
            g = 0.5
            for i in range(n):
                x = acc_l[i]
                bv = buf[idx]
                y = -g * x + bv
                buf[idx] = x + g * bv
                acc_l[i] = y
                idx += 1
                if idx >= ln:
                    idx = 0
            self.apidx[ai] = idx
        for ai in range(2, 4):
            buf = self.allpass[ai]
            ln = len(buf) - 1
            idx = self.apidx[ai]
            g = 0.5
            for i in range(n):
                x = acc_r[i]
                bv = buf[idx]
                y = -g * x + bv
                buf[idx] = x + g * bv
                acc_r[i] = y
                idx += 1
                if idx >= ln:
                    idx = 0
            self.apidx[ai] = idx
        m = mix
        out = np.empty_like(dry)
        out[:, 0] = dry[:, 0] * (1 - m) + acc_l * m
        out[:, 1] = dry[:, 1] * (1 - m) + acc_r * m
        return out
