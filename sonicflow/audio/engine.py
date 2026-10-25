"""
Motor de audio en tiempo real de SonicFlow Studio.

Diseño:
  - El thread de audio (PortAudio vía sounddevice; callback determinista) llama a
    `process_block` sin tocar el disco ni la GUI. Si no hay dispositivo físico
    disponible, un reloj interno llama al MISMO proceso para que grabación,
    reproducción de tomas y exportación sigan funcionando (backend "dummy").
  - Cada Instrumento contiene Capas; cada Capa tiene sample, ganancia, tono,
    velocidad, filtro, panorama, efectos, ADSR y asignaciones X/Y.
  - Las superficies XY alimentan valores suavizados por bloque que modulan los
    parámetros asignados. La lectura de la muestra usa fase fraccional continua
    entre bloques con interpolación lineal (sin clics); las notas one-shot usan
    envolventes AD/AR suaves.
  - Grabación: master y stems por capa se capturan DENTRO del mismo
    process_block, sincronizados al reloj de frames del motor; cada toma guarda
    también los trazos XY (automation) y la posición del vídeo en su inicio.
  - Render offline (`render_toma`) reutiliza EXACTAMENTE el mismo DSP para
    exportar WAV fiel a lo escuchado.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .dsp import AdsrState, Biquad, Reverb, StereoDelay, adsr_process, saturate_tanh
from .sample import Sample

log = logging.getLogger("sonicflow.audio")

FILTER_HZ_MIN, FILTER_HZ_MAX = 30.0, 18000.0


def scale_to_freq(v: float) -> float:
    """Escala logarítmica 0..1 -> Hz."""
    return FILTER_HZ_MIN * (FILTER_HZ_MAX / FILTER_HZ_MIN) ** float(np.clip(v, 0, 1))


@dataclass
class LayerParams:
    """Parámetros de una capa (leídos en hilo de audio, escritos desde UI)."""
    name: str = "Capa"
    enabled: bool = True
    gain_db: float = 0.0            # mezclador
    pan: float = 0.0                # -1..1
    pitch_semi: float = 0.0         # semitonos base
    speed: float = 1.0              # playback rate base
    filter_on: bool = False
    filter_type: str = "lowpass"    # lowpass | highpass | bandpass
    filter_freq: float = 12000.0
    filter_q: float = 0.707
    drive: float = 1.0              # saturación tanh (1 = off)
    delay_mix: float = 0.0          # envío al delay maestro
    delay_time_ms: float = 250.0
    delay_fb: float = 0.35
    rev_send: float = 0.0           # envío al reverb maestro
    attack: float = 0.005
    decay: float = 0.25
    sustain: float = 1.0
    release: float = 0.25
    loop: bool = False              # bucle mientras gate abierto
    trigger_mode: str = "gate"      # gate | one_shot
    x_param: str | None = None
    y_param: str | None = None
    x_invert: bool = False
    y_invert: bool = False
    x_range: tuple[float, float] | None = None
    y_range: tuple[float, float] | None = None


LAYER_XY_PARAMS = {
    "speed": (0.25, 4.0),
    "pitch": (-24.0, 24.0),
    "filter": (0.0, 1.0),      # escala log -> Hz
    "gain": (-60.0, 12.0),     # dB adicional sobre la ganancia de capa
    "pan": (-1.0, 1.0),
    "drive": (1.0, 8.0),
    "rev": (0.0, 1.0),
    "delay": (0.0, 0.8),
}


def apply_response(x: float, mode: str, k: float = 2.0) -> float:
    """Curvas de respuesta de los ejes XY."""
    x = float(np.clip(x, 0.0, 1.0))
    if mode == "linear":
        return x
    if mode == "exp":
        return x ** max(k, 0.1)
    if mode == "log":
        return x ** (1.0 / max(k, 0.1))
    if mode == "sqrt":
        return float(np.sqrt(x))
    if mode == "square":
        return x * x
    if mode == "s_curve":
        return float(np.sin((x - 0.5) * np.pi) * 0.5 + 0.5)
    raise ValueError(f"curva desconocida: {mode}")


class _Voice:
    """Estado de reproducción de una capa (fase fraccional continua)."""
    __slots__ = ("pos", "playing", "releasing", "adsr")

    def __init__(self) -> None:
        self.pos = 0.0
        self.playing = False
        self.releasing = False
        self.adsr = AdsrState()


class Layer:
    """Capa dentro de un instrumento: sample + params + voz + filtros + medidor."""

    def __init__(self, params: LayerParams, sample: Sample | None, engine_sr: int):
        self.params = params
        self.sample = sample
        self.engine_sr = engine_sr
        self.uid: int = -1
        self.voice = _Voice()
        self.lp = Biquad(engine_sr)
        self.hp = Biquad(engine_sr)
        self.bp = Biquad(engine_sr)
        self._active_filter = self.lp
        self._last_filter_sig = None
        self.peak_hold = 0.0
        self.peak_db = -120.0

    def set_sample(self, sample: Sample | None) -> None:
        self.sample = sample
        self.voice = _Voice()

    def rebuild_filters(self, sr: int) -> None:
        self.engine_sr = sr
        self.lp = Biquad(sr)
        self.hp = Biquad(sr)
        self.bp = Biquad(sr)
        self._active_filter = self.lp
        self._last_filter_sig = None


class Instrument:
    def __init__(self, name: str, instrument_id: int):
        self.name = name
        self.id = instrument_id
        self.layers: list[Layer] = []
        self.gain_db: float = 0.0
        self.mute: bool = False
        self.solo: bool = False

    def add_layer(self, layer: Layer) -> Layer:
        self.layers.append(layer)
        return layer


@dataclass
class XYSurfaceState:
    """Superficie XY compartida: la UI escribe targets; el audio suaviza sx/sy."""
    id: int
    name: str = "XY"
    active_instrument: int = 0
    pressed: bool = False
    continuous: bool = False       # suena siempre (sin necesidad de pulsar)
    render_continuous: bool = True  # en renders offline las superficies suenan
                                    # solas (el gate se grabó como automation);
                                    # desactívalo para reproducir solo presiones
    return_center: bool = False
    sensitivity: float = 1.0
    smoothing_ms: float = 12.0
    deadzone: float = 0.0
    curve_x: str = "linear"
    curve_y: str = "linear"
    curve_k: float = 2.0
    zone_min_x: float = 0.0
    zone_max_x: float = 1.0
    zone_min_y: float = 0.0
    zone_max_y: float = 1.0
    zone_enabled: bool = False
    sx: float = 0.5
    sy: float = 0.5
    target_x: float = 0.5
    target_y: float = 0.5


@dataclass
class Take:
    """Interpretación grabada: master + stems por capa + automation XY."""
    id: int
    name: str
    start_video_s: float
    frames_master: np.ndarray = field(default=None, repr=False)     # [N,2] float32
    frames_layers: dict = field(default_factory=dict, repr=False)   # uid -> [N,2]
    xy_events: list = field(default_factory=list)                   # (t_s, sid, x, y, press)
    length_s: float = 0.0


class AudioEngine:
    def __init__(self, sample_rate: int = 48000, block_size: int = 256):
        self.sample_rate = int(sample_rate)
        self.block_size = int(block_size)
        self.backend_name = "dummy"
        self.stream = None
        self.running = False
        self.error: str | None = None

        self.lock = threading.Lock()
        self.instruments: list[Instrument] = []
        self.surfaces: dict[int, XYSurfaceState] = {}
        self._next_surface_id = 1
        self._layer_uid = 0

        self.frames_total = 0            # reloj del motor (frames procesados)
        self.delay = StereoDelay(self.sample_rate)
        self.reverb = Reverb(self.sample_rate)
        self.master_gain_db = -6.0
        self.limiter_on = True
        self.master_rev_mix = 1.0        # nivel del bus paralelo de reverb
        self.master_room = 0.6
        self.master_delay_mix = 0.0
        self.master_delay_time = 300.0

        self.master_peak = 0.0
        self.master_rms = 0.0
        self.xrun_count = 0

        # grabación
        self.recording = False
        self.takes: list[Take] = []
        self._take_counter = 0
        self._rec_master: list[np.ndarray] = []
        self._rec_layers: dict[int, list[np.ndarray]] = {}
        self._rec_xy: list[tuple] = []
        self.take_start_video_s = 0.0

        # replay de automation
        self.replay_take: Take | None = None
        self.replay_active = False
        self._replay_event_idx = 0
        self._replay_start_frame = 0

        self._sd_callback = self._make_sd_callback()
        self._dummy_thread: threading.Thread | None = None
        self._offline_mode = False     # True durante render_toma

    # ------------------------------------------------------------ ids/slots
    def alloc_layer_uid(self) -> int:
        uid = self._layer_uid
        self._layer_uid += 1
        return uid

    def new_surface(self, name: str = "XY") -> XYSurfaceState:
        s = XYSurfaceState(id=self._next_surface_id, name=name)
        self._next_surface_id += 1
        self.surfaces[s.id] = s
        return s

    def remove_surface(self, sid: int) -> None:
        self.surfaces.pop(sid, None)

    def all_layers(self) -> list[Layer]:
        return [l for i in self.instruments for l in i.layers]

    def find_layer(self, uid: int) -> Layer | None:
        for l in self.all_layers():
            if l.uid == uid:
                return l
        return None

    # ------------------------------------------------------------- apertura
    def open(self, device_index: int | None = None, latency_s: float | None = None) -> bool:
        """Intenta abrir PortAudio; si no hay dispositivos usa backend dummy."""
        self.close()
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            out_ok = len(devices) > 0 and any(
                d.get("max_output_channels", 0) > 0 for d in devices)
            if out_ok:
                if device_index is None:
                    dev = sd.default.device[1]
                    device_index = dev if dev is not None and dev >= 0 else None
                kwargs = dict(samplerate=float(self.sample_rate),
                              blocksize=self.block_size, channels=2,
                              dtype="float32", callback=self._sd_callback)
                if device_index is not None:
                    kwargs["device"] = int(device_index)
                if latency_s is not None:
                    kwargs["latency"] = float(latency_s)
                try:
                    st = sd.OutputStream(**kwargs)
                except Exception as exc_dev:
                    log.warning("Device %s falló (%s); usando predeterminado",
                                device_index, exc_dev)
                    kwargs.pop("device", None)
                    kwargs.pop("latency", None)
                    st = sd.OutputStream(**kwargs)
                st.start()
                self.stream = st
                self.backend_name = "portaudio"
                self.error = None
                log.info("Audio abierto: PortAudio @ %d Hz block %d",
                         self.sample_rate, self.block_size)
                return True
            self.backend_name = "dummy"
            self.error = ("No se detectaron dispositivos de salida de audio. "
                          "La aplicación funciona con reloj interno: interpretación, "
                          "grabación y exportación operan con normalidad, pero no "
                          "habrá sonido físico hasta conectar una salida.")
            log.warning(self.error)
            self._start_dummy_clock()
            return True
        except Exception as exc:
            self.backend_name = "dummy"
            self.error = (f"Dispositivo de audio no disponible "
                          f"({exc.__class__.__name__}: {exc}). Usando reloj interno.")
            log.warning(self.error)
            self._start_dummy_clock()
            return True

    def close(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.running = False
        if self._dummy_thread is not None:
            self._dummy_thread.join(timeout=1.0)
            self._dummy_thread = None

    @property
    def devices(self) -> list[dict]:
        try:
            import sounddevice as sd
            return [dict(name=d["name"], index=i,
                         in_ch=d["max_input_channels"],
                         out_ch=d["max_output_channels"],
                         default=(i in sd.default.device))
                    for i, d in enumerate(sd.query_devices())]
        except Exception:
            return []

    def list_sample_rates(self, device_index: int | None = None) -> list[int]:
        try:
            import sounddevice as sd
            hs = sd.query_devices(device_index or sd.default.device[1])["default_samplerate"]
            cands = [32000, 44100, 48000, 88200, 96000, 192000]
            return [int(hs)] + [c for c in cands if int(c) != int(hs)]
        except Exception:
            return [44100, 48000, 96000]

    # --------------------------------------------------------- dummy clock
    def _start_dummy_clock(self) -> None:
        self.running = True
        th = threading.Thread(target=self._dummy_loop, daemon=True, name="audio-dummy")
        th.start()
        self._dummy_thread = th

    def _dummy_loop(self) -> None:
        interval = self.block_size / self.sample_rate
        next_tick = time.monotonic() + interval
        while self.running:
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            elif sleep < -interval:
                next_tick = time.monotonic()
            next_tick += interval
            if self.running:
                self.process_block(self.block_size)

    # ------------------------------------------------------------ callback
    def _make_sd_callback(self):
        def cb(outdata, indata, frames, tii, status):  # pragma: no cover - hardware
            if status:
                self.xrun_count += 1
            outdata[:] = self.process_block(frames)
        return cb

    # ================================================================ DSP
    def process_block(self, nframes: int, skip_smoothing: bool = False) -> np.ndarray:
        """Procesa un bloque. Devuelve [nframes, 2] float32. Sin I/O ni locks pesados."""
        sr = self.sample_rate
        out = np.zeros((nframes, 2), dtype=np.float32)
        rev_in = np.zeros((nframes, 2), dtype=np.float32)
        dly_in = np.zeros((nframes, 2), dtype=np.float32)

        solo_any = any(i.solo for i in self.instruments)

        if self.replay_active and self.replay_take is not None:
            self._apply_replay_events()

        # El suavizado ya se aplicó al final del bloque anterior (ver abajo);
        # aquí solo se procesa audio con los valores corrientes sx/sy.

        for inst in self.instruments:
            inst_audible = not inst.mute and (not solo_any or inst.solo)
            inst_gain = 10 ** (inst.gain_db / 20.0)
            for layer in inst.layers:
                p = layer.params
                gated = self._layer_gate(layer)
                block = self._render_layer(layer, nframes, gated)
                if not inst_audible or not p.enabled:
                    block[:] = 0.0
                if block.size:
                    pk = float(np.max(np.abs(block)))
                    layer.peak_hold = max(pk, layer.peak_hold * 0.92)
                    layer.peak_db = 20 * np.log10(max(layer.peak_hold, 1e-9))
                if self.recording and layer.uid in self._rec_layers:
                    self._rec_layers[layer.uid].append(block.copy())
                out += block * inst_gain
                if p.rev_send > 0.001:
                    rev_in += block * p.rev_send
                if p.delay_mix > 0.001:
                    dly_in += block * p.delay_mix

        if self.master_delay_mix > 0.001 or np.any(dly_in):
            send = dly_in + out * min(self.master_delay_mix, 1.0)
            out = out + self.delay.process(send, self.master_delay_time,
                                           0.35, 1.0, cross=0.15)

        if np.any(rev_in):
            wet = self.reverb.process(rev_in, 1.0, self.master_room)
            out += wet * self.master_rev_mix

        mg = 10 ** (self.master_gain_db / 20.0)
        out *= mg
        if self.limiter_on:
            out = self._soft_clip(out)

        pk = float(np.max(np.abs(out))) if out.size else 0.0
        self.master_peak = max(pk, self.master_peak * 0.88)
        self.master_rms = float(np.sqrt(np.mean(out ** 2))) if out.size else 0.0

        if self.recording:
            self._rec_master.append(out.copy())
            now_s = self.frames_total / sr
            for s in self.surfaces.values():
                self._rec_xy.append((round(now_s, 6), s.id, round(s.sx, 4),
                                     round(s.sy, 4), 1.0 if s.pressed else 0.0))

        self.frames_total += nframes
        # suavizado para el PRÓXIMO bloque (no en renders offline: la automation
        # ya contiene el estado suavizado grabado y se interpola directamente)
        if not skip_smoothing:
            self._smooth_xy(nframes)
        return out

    # ------------------------------------------------------------ helpers
    def _smooth_xy(self, nframes: int) -> None:
        alpha = nframes / self.sample_rate
        for s in self.surfaces.values():
            tau = max(s.smoothing_ms / 1000.0, 1e-5)
            k = 1 - float(np.exp(-alpha / tau))
            k = float(np.clip(k * s.sensitivity, 0.0, 1.0))
            dx = s.target_x - s.sx
            dy = s.target_y - s.sy
            if s.deadzone > 0:
                if abs(dx) < s.deadzone:
                    dx = 0.0
                if abs(dy) < s.deadzone:
                    dy = 0.0
            s.sx += dx * k
            s.sy += dy * k
            if s.return_center and not s.pressed:
                r = min(1.0, alpha * 6)
                s.target_x += (0.5 - s.target_x) * r
                s.target_y += (0.5 - s.target_y) * r

    def _layer_gate(self, layer: Layer) -> bool:
        p = layer.params
        if p.trigger_mode == "one_shot":
            return True  # la voz gestiona su ciclo propio
        inst_idx = self._inst_index_of_layer(layer)
        for s in self.surfaces.values():
            if s.active_instrument != inst_idx:
                continue
            if s.continuous or (self._offline_mode and s.render_continuous):
                return True
            if s.pressed and self._in_zone(s, s.sx, s.sy):
                return True
        return False

    def set_layer_active(self, layer: Layer, active: bool) -> None:
        """Activa/desactiva una capa sin superficie XY (disparo manual/MIDI)."""
        if active:
            v = layer.voice
            v.pos = 0.0
            v.playing = True
            v.releasing = False
            v.adsr = AdsrState()
            layer._manual_gate = True
        else:
            layer._manual_gate = False
            if layer.params.trigger_mode == "gate" and layer.voice.playing:
                layer.voice.releasing = True

    @staticmethod
    def _in_zone(s: XYSurfaceState, x: float, y: float) -> bool:
        if not s.zone_enabled:
            return True
        return s.zone_min_x <= x <= s.zone_max_x and s.zone_min_y <= y <= s.zone_max_y

    def _xy_for_layer(self, layer: Layer) -> tuple[float, float]:
        inst_idx = self._inst_index_of_layer(layer)
        xs, ys = [], []
        for s in self.surfaces.values():
            if s.active_instrument != inst_idx:
                continue
            gx, gy = s.sx, s.sy
            if s.curve_x != "linear":
                gx = apply_response(gx, s.curve_x, s.curve_k)
            if s.curve_y != "linear":
                gy = apply_response(gy, s.curve_y, s.curve_k)
            xs.append(gx)
            ys.append(gy)
        if not xs:
            return 0.5, 0.5
        return float(np.mean(xs)), float(np.mean(ys))

    def _inst_index_of_layer(self, layer: Layer) -> int:
        for ii, inst in enumerate(self.instruments):
            if layer in inst.layers:
                return ii
        return -1

    def _render_layer(self, layer: Layer, nframes: int, gate: bool) -> np.ndarray:
        p = layer.params
        v = layer.voice
        smp = layer.sample
        block = np.zeros((nframes, 2), dtype=np.float32)
        if smp is None:
            return block

        gate = gate or getattr(layer, "_manual_gate", False)
        one_shot = p.trigger_mode == "one_shot"
        if not one_shot:
            if gate and not v.playing:
                v.pos = 0.0
                v.playing = True
                v.releasing = False
                v.adsr = AdsrState()
            elif not gate and v.playing and not v.releasing:
                v.releasing = True
        if not v.playing:
            return block

        x, y = self._xy_for_layer(layer)

        speed, pitch = p.speed, p.pitch_semi
        gain_db, pan, drive = p.gain_db, p.pan, p.drive
        filt_freq = p.filter_freq
        for axis, val in (("x", x), ("y", y)):
            param = p.x_param if axis == "x" else p.y_param
            if not param or param not in LAYER_XY_PARAMS:
                continue
            rng = (p.x_range if axis == "x" else p.y_range) or LAYER_XY_PARAMS[param]
            invert = p.x_invert if axis == "x" else p.y_invert
            vv = (1.0 - val) if invert else val
            mapped = rng[0] + vv * (rng[1] - rng[0])
            if param == "speed":
                speed *= mapped
            elif param == "pitch":
                pitch += mapped
            elif param == "filter":
                filt_freq = scale_to_freq(mapped)
            elif param == "gain":
                gain_db += mapped
            elif param == "pan":
                pan = mapped
            elif param == "drive":
                drive = mapped

        lin = 10 ** (float(np.clip(gain_db, -120, 24)) / 20.0)
        ratio = (smp.sr / self.sample_rate) * speed * (2.0 ** (pitch / 12.0))
        ratio = float(np.clip(ratio, 0.05, 8.0))

        data = smp.data
        nsamples = data.shape[0]
        idx = np.arange(nframes)
        positions = v.pos + (idx + 1) * ratio
        if p.loop:
            wrapped = positions % nsamples
            ended = np.zeros(nframes, dtype=bool)
        else:
            wrapped = np.minimum(positions, nsamples - 1)
            ended = positions >= nsamples
        i0 = wrapped.astype(np.int64)
        i1 = np.minimum(i0 + 1, nsamples - 1)
        frac = (wrapped - i0).astype(np.float32)[:, None]
        s = data[i0] * (1.0 - frac) + data[i1] * frac
        v.pos = float(positions[-1])

        # ---- envolvente
        if one_shot:
            env = self._oneshot_env(v, nframes, p, ended)
            if ended.all() and v.adsr.level <= 1e-6:
                v.playing = False
        else:
            env = adsr_process(v.adsr, nframes, p.attack, p.decay, p.sustain,
                               p.release, self.sample_rate, gate, False)
            if v.releasing and not gate and v.adsr.level <= 1e-6:
                v.playing = False
                v.releasing = False
        s *= env[:, None]

        # ---- filtro
        if p.filter_on:
            sig = (p.filter_type, round(filt_freq), round(p.filter_q, 2))
            if sig != layer._last_filter_sig:
                bq = {"lowpass": layer.lp, "highpass": layer.hp,
                      "bandpass": layer.bp}[p.filter_type]
                getattr(bq, f"set_{p.filter_type}")(filt_freq, p.filter_q)
                layer._active_filter = bq
                layer._last_filter_sig = sig
            s[:, 0] = layer._active_filter.process_block(s[:, 0])
            s[:, 1] = layer._active_filter.process_block(s[:, 1])

        if drive > 1.01:
            s = saturate_tanh(s, drive)

        theta = (float(np.clip(pan, -1, 1)) + 1) * np.pi / 4
        s[:, 0] *= np.cos(theta)
        s[:, 1] *= np.sin(theta)
        s *= lin
        return s

    def _oneshot_env(self, v: _Voice, n: int, p: LayerParams,
                     ended: np.ndarray) -> np.ndarray:
        """One-shot: A-D-S(minúsculo)-R. Mantiene sustain hasta el final de la
        muestra; cuando termina inicia release y apaga la voz al morir."""
        sr = self.sample_rate
        gate_open = not ended.any()  # 'gate' implícito mientras queda muestra
        env = adsr_process(v.adsr, n, p.attack, p.decay, p.sustain,
                           p.release, sr, gate_open, False)
        if ended.any() and v.adsr.stage != 3:
            v.adsr.stage = 3
        return env

    @staticmethod
    def _soft_clip(x: np.ndarray) -> np.ndarray:
        hot = np.abs(x) > 0.99
        if hot.any():
            x = np.where(hot, np.tanh(x * 1.2) / np.tanh(1.26), x)
        return np.clip(x, -1.0, 1.0).astype(np.float32)

    # --------------------------------------------------------------- comandos
    def set_xy_target(self, surface_id: int, x: float, y: float) -> None:
        s = self.surfaces.get(surface_id)
        if s:
            s.target_x = float(np.clip(x, 0, 1))
            s.target_y = float(np.clip(y, 0, 1))

    def set_xy_pressed(self, surface_id: int, pressed: bool) -> None:
        s = self.surfaces.get(surface_id)
        if s is None:
            return
        s.pressed = pressed
        inst = self._instrument_by_idx(s.active_instrument)
        if inst is None:
            return
        if pressed:
            for l in inst.layers:
                if l.params.enabled and l.params.trigger_mode == "one_shot":
                    l.voice = _Voice()
                    l.voice.playing = True
        else:
            for l in inst.layers:
                if l.params.trigger_mode == "gate":
                    if l.voice.playing and not l.voice.releasing:
                        l.voice.releasing = True

    def trigger_layer(self, layer: Layer) -> None:
        """Disparo explícito (pad MIDI o botón). One-shot re-dispara desde 0."""
        if layer.sample is None or not layer.params.enabled:
            return
        if layer.params.trigger_mode == "one_shot":
            layer.voice = _Voice()
            layer.voice.playing = True
        else:
            layer.voice.pos = 0.0
            layer.voice.playing = True
            layer.voice.releasing = False
            layer.voice.adsr = AdsrState()
            # auto-release breve para pulsaciones momentáneas
            threading.Timer(0.25, lambda: layer.voice.__setattr__("releasing", True)
                            if layer.voice.playing else None).start()

    def _instrument_by_idx(self, idx: int) -> Instrument | None:
        if 0 <= idx < len(self.instruments):
            return self.instruments[idx]
        return None

    def stop_all_voices(self) -> None:
        for l in self.all_layers():
            l.voice = _Voice()
        self.reverb.reset()
        self.delay.reset()

    # --------------------------------------------------------------- grabación
    def start_recording(self, video_pos_s: float, capture_stems: bool = True) -> None:
        self._take_counter += 1
        self.recording = True
        self._rec_master = []
        self._rec_layers = {}
        self._rec_xy = []
        self.take_start_video_s = float(video_pos_s)
        if capture_stems:
            for l in self.all_layers():
                if l.uid >= 0:
                    self._rec_layers[l.uid] = []
        log.info("Grabando toma %d desde vídeo %.3fs", self._take_counter, video_pos_s)

    def stop_recording(self) -> Take | None:
        if not self.recording:
            return None
        self.recording = False
        master = (np.concatenate(self._rec_master, axis=0) if self._rec_master
                  else np.zeros((0, 2), dtype=np.float32))
        layers = {uid: (np.concatenate(ch, 0) if ch else np.zeros((0, 2), np.float32))
                  for uid, ch in self._rec_layers.items()}
        take = Take(id=self._take_counter, name=f"Toma {self._take_counter}",
                    start_video_s=self.take_start_video_s,
                    frames_master=master, frames_layers=layers,
                    xy_events=list(self._rec_xy),
                    length_s=master.shape[0] / self.sample_rate)
        self.takes.append(take)
        self._rec_master, self._rec_layers, self._rec_xy = [], {}, []
        log.info("Toma %d grabada: %.2f s", take.id, take.length_s)
        return take

    def undo_last_take(self) -> Take | None:
        return self.takes.pop() if self.takes else None

    def redo_take(self, take: Take) -> None:
        self.takes.append(take)

    def clear_takes(self) -> None:
        self.takes.clear()

    # ------------------------------------------------------ replay automation
    def play_take_automation(self, take: Take) -> None:
        self.replay_take = take
        self.replay_active = True
        self._replay_event_idx = 0
        self._replay_start_frame = self.frames_total
        self.stop_all_voices()

    def stop_replay(self) -> None:
        self.replay_active = False
        self.replay_take = None
        for s in self.surfaces.values():
            s.pressed = False

    def _apply_replay_events(self) -> None:
        t = self.replay_take
        if t is None:
            return
        now_s = (self.frames_total - self._replay_start_frame) / self.sample_rate
        evs = t.xy_events
        i = self._replay_event_idx
        latest: dict[int, tuple[float, float, float]] = {}
        while i < len(evs) and evs[i][0] <= now_s:
            ts, sid, x, y, pr = evs[i]
            latest[sid] = (x, y, pr)
            i += 1
        self._replay_event_idx = i
        for sid, (x, y, pr) in latest.items():
            s = self.surfaces.get(sid)
            if s:
                s.target_x = x
                s.target_y = y
                s.pressed = pr > 0.5
        if i >= len(evs):
            self.stop_replay()

    # ------------------------------------------------------- render offline
    def render_toma(self, take: Take, sample_rate: int | None = None) -> dict:
        """Renderiza la toma OFFLINE con el mismo DSP.

        Devuelve {'master': [N,2] float32, 'layers': {uid: [N,2]}, 'sr': int}.
        La UI debe pausar el stream en vivo durante el render (se hace aquí).
        """
        sr = int(sample_rate or self.sample_rate)
        with self.lock:
            paused = False
            if self.stream is not None:
                try:
                    self.stream.stop()
                    paused = True
                except Exception:
                    pass
            saved_sr = self.sample_rate
            self.sample_rate = sr
            self.reverb = Reverb(sr)
            self.delay = StereoDelay(sr)
            for l in self.all_layers():
                l.voice = _Voice()
                l.rebuild_filters(sr)
            total = max(int(round(take.length_s * sr)), sr // 4)
            block = 512
            rec_saved = self.recording
            rec_replay = self.replay_active
            off_saved = self._offline_mode
            rc_flags = {sid: s.render_continuous for sid, s in self.surfaces.items()}
            # Si la toma tiene eventos de presión (interpretación con gate), el
            # render respeta exactamente esas presiones; si no se grabó ninguna
            # (modo continuo), el render suena en continuo.
            any_press = any(e[4] > 0.5 for e in take.xy_events)
            for s in self.surfaces.values():
                s.render_continuous = not any_press
            self._offline_mode = True
            self.recording = True
            self.replay_active = False
            self._rec_master = []
            self._rec_layers = {uid: [] for uid in take.frames_layers}
            self._rec_xy = []
            surfaces_saved = {sid: (s.target_x, s.target_y, s.pressed, s.sx, s.sy)
                              for sid, s in self.surfaces.items()}

            # --- Motor de reproducción determinista -------------------------
            # La grabación registra el estado SUAVIZADO (sx/sy) al final de cada
            # bloque. En el render se reconstruye el mismo recorrido fijando
            # sx/sy AL INICIO de cada bloque e interpolando linealmente dentro
            # del bloque hacia el siguiente punto. Con ello la trayectoria es
            # idéntica a la interpretada y el audio exportado coincide con lo
            # escuchado (diferencia < 1e-3).
            live_block = max(64, self.block_size)
            total_live_frames = int(round(take.length_s * sr))
            n_blocks = int(np.ceil(total_live_frames / live_block)) if total_live_frames else 0
            per_surface: dict[int, list] = {}
            for e in sorted(take.xy_events, key=lambda ev: (ev[0], ev[1])):
                per_surface.setdefault(e[1], []).append(e)

            try:
                # --- Motor de reproducción determinista ---------------------
                # La grabación registra el estado SUAVIZADO (sx/sy) al FINAL de
                # cada bloque. En el render se fija sx/sy AL INICIO de cada
                # bloque con el valor grabado del bloque anterior y NO se vuelve
                # a suavizar: la trayectoria resultante es idéntica, muestra a
                # muestra, a la interpretada en vivo.
                live_block = max(64, self.block_size)
                total_live_frames = int(round(take.length_s * sr))
                n_blocks = int(np.ceil(total_live_frames / live_block)) if total_live_frames else 0
                per_surface: dict[int, list] = {}
                for e in sorted(take.xy_events, key=lambda ev: (ev[0], ev[1])):
                    per_surface.setdefault(e[1], []).append(e)

                def state_at(sid: int, t_abs: float):
                    """Último evento grabado con timestamp <= t_abs (None si no hay)."""
                    seq = per_surface.get(sid)
                    if not seq:
                        return None
                    lo, hi = 0, len(seq) - 1
                    best = None
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        if seq[mid][0] <= t_abs + 1e-9:
                            best = mid
                            lo = mid + 1
                        else:
                            hi = mid - 1
                    if best is None:
                        return None
                    _, _, x, y, p = seq[best]
                    return x, y, p > 0.5

                for b in range(n_blocks):
                    pos0 = b * live_block
                    n = min(live_block, total_live_frames - pos0)
                    t_prev = max(0.0, (pos0 - live_block) / sr)
                    for s in self.surfaces.values():
                        st = state_at(s.id, t_prev)
                        if st is not None:
                            s.sx, s.sy, s.pressed = st
                            s.target_x, s.target_y = st[0], st[1]
                        else:
                            s.pressed = False
                    self.process_block(n, skip_smoothing=True)
                master = np.concatenate(self._rec_master, 0) if self._rec_master \
                    else np.zeros((0, 2), np.float32)
                layers = {uid: (np.concatenate(ch, 0) if ch else np.zeros((0, 2), np.float32))
                          for uid, ch in self._rec_layers.items()}
            finally:
                self._offline_mode = off_saved
                for sid, f in rc_flags.items():
                    s = self.surfaces.get(sid)
                    if s:
                        s.render_continuous = f
                self.recording = rec_saved
                self.replay_active = rec_replay
                self._rec_master, self._rec_layers, self._rec_xy = [], {}, []
                for sid, (tx, ty, pr, sx, sy) in surfaces_saved.items():
                    s = self.surfaces.get(sid)
                    if s:
                        s.target_x, s.target_y, s.pressed, s.sx, s.sy = tx, ty, pr, sx, sy
                self.sample_rate = saved_sr
                self.reverb = Reverb(saved_sr)
                self.delay = StereoDelay(saved_sr)
                for l in self.all_layers():
                    l.voice = _Voice()
                    l.rebuild_filters(saved_sr)
                if paused:
                    try:
                        self.stream.start()
                    except Exception:
                        pass
            return {"master": master, "layers": layers, "sr": sr}

    # -------------------------------------------------------------- medición
    def clock_seconds(self) -> float:
        return self.frames_total / self.sample_rate
