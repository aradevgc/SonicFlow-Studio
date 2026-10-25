#!/usr/bin/env python3
"""
Prueba funcional del motor de audio de SonicFlow Studio (sin GUI).

Demuestra el flujo completo de la Fase 1 + Fase 2 con el código real:
  1. Carga las muestras de demostración (resources/demosounds/*.wav).
  2. Crea un instrumento con capas y asigna X=speed / Y=gain a una superficie XY.
  3. Simula una interpretación moviendo la superficie XY.
  4. Graba la toma (master + stems + automation) DENTRO del process_block.
  5. Exporta WAV 24-bit con render offline determinista (mismo DSP).

Uso:
    python scripts/demo_engine.py [salida.wav]

Si tienes salida de audio física, añade --play para escucharla en tiempo real.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from sonicflow.audio.engine import AudioEngine, Instrument, Layer, LayerParams  # noqa: E402
from sonicflow.audio.sample import Sample  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(ROOT, "resources", "demosounds")


def make_layer(engine: AudioEngine, name: str, wav: str, **overrides) -> Layer:
    p = LayerParams(name=name, **overrides)
    path = os.path.join(DEMO_DIR, wav)
    smp = Sample.from_file(path) if os.path.isfile(path) else None
    layer = Layer(p, smp, engine.sample_rate)
    layer.uid = engine.alloc_layer_uid()
    return layer


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
        else os.path.join(ROOT, "examples", "demo_take.wav")
    play = "--play" in sys.argv

    engine = AudioEngine(sample_rate=48000, block_size=256)
    opened = engine.open()
    print(f"Backend de audio: {engine.backend_name} ({'abierto' if opened else 'fallo'})")
    if play and engine.backend_name == "portaudio":
        # modo interpretación en tiempo real: el hilo de audio toca solo;
        # aquí sólo movemos el objetivo XY y grabamos hasta pulsar Enter.
        input("Pulsa Enter para INICIAR la grabación en tiempo real ")

    # --- instrumento cinematográfico de ejemplo --------------------------- #
    inst = Instrument("Movimiento cinematográfico", 0)
    air = make_layer(engine, "Aire", "air_whoosh.wav",
                     loop=True, trigger_mode="gate", x_param="speed", y_param="gain",
                     rev_send=0.3)
    boom = make_layer(engine, "Golpe grave", "deep_impact.wav",
                      trigger_mode="one_shot", y_param="pitch", gain_db=-3)
    metal = make_layer(engine, "Textura metálica", "metal_texture.wav",
                       loop=True, x_param="filter", y_param="pan",
                       filter_on=True, filter_freq=8000, gain_db=-6)
    amb = make_layer(engine, "Ambiente", "ambient_pad.wav",
                     loop=True, y_param="rev", gain_db=-12, rev_send=0.6)
    for l in (air, boom, metal, amb):
        inst.add_layer(l)
    engine.instruments.append(inst)

    surf = engine.new_surface("XY principal")
    surf.active_instrument = 0
    surf.continuous = True          # suena sin necesidad de pulsar (modo performance)
    surf.smoothing_ms = 10.0

    # --- interpretación simulada: barrido XY ------------------------------- #
    if play and engine.backend_name == "portaudio":
        # Modo tiempo real: el hilo de audio procesa solo; movemos el objetivo
        # XY por cronometría y dejamos que el usuario pulse Enter para parar.
        import time as _time
        dur_s = 8.0
        engine.start_recording(video_pos_s=0.0, capture_stems=True)
        t0 = _time.perf_counter()
        print("Interpretando en tiempo real (barrido XY de 8 s)...")
        while _time.perf_counter() - t0 < dur_s:
            t = (_time.perf_counter() - t0) / dur_s
            engine.set_xy_target(surf.id, float(np.clip(t, 0, 1)),
                                 float(np.clip(np.sin(t * np.pi), 0, 1)))
            _time.sleep(0.004)
    else:
        # Modo offline (sin dispositivo físico): procesamos bloques a mano.
        dur_s = 4.0
        n_blocks = int(dur_s * engine.sample_rate // engine.block_size)
        engine.start_recording(video_pos_s=0.0, capture_stems=True)
        for i in range(n_blocks):
            t = i / n_blocks
            engine.set_xy_target(surf.id, float(np.clip(t, 0, 1)),         # X: velocidad
                                 float(np.clip(np.sin(t * np.pi), 0, 1)))   # Y: intensidad
            engine.process_block(engine.block_size)
    take = engine.stop_recording()
    engine.close()

    assert take is not None and take.frames_master is not None
    peak = float(np.max(np.abs(take.frames_master)))
    print(f"Toma '{take.name}': {take.length_s:.2f} s, eventos XY: {len(take.xy_events)}, "
          f"stems: {len(take.frames_layers)}, pico master: {peak:.3f}")
    assert peak > 0.01, "¡La grabación está en silencio! El motor no produjo audio."

    # --- exportación WAV 24-bit -------------------------------------------- #
    import soundfile as sf
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sf.write(out, take.frames_master, engine.sample_rate, subtype="PCM_24")
    print(f"OK: mezcla estéreo 24-bit exportada a {out}")

    stem_dir = os.path.splitext(out)[0] + "_stems"
    os.makedirs(stem_dir, exist_ok=True)
    for uid, data in take.frames_layers.items():
        layer = engine.find_layer(uid)
        name = layer.params.name.replace(" ", "_") if layer else f"layer_{uid}"
        p = os.path.join(stem_dir, f"{name}.wav")
        sf.write(p, data, engine.sample_rate, subtype="PCM_24")
        print(f"OK: stem -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
