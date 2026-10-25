"""Pruebas básicas del motor de audio y de sincronización (Fase 1-2)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sonicflow.audio.engine import AudioEngine, Instrument, Layer, LayerParams
from sonicflow.audio.sample import Sample

DEMO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "resources", "demosounds")


def build_engine():
    e = AudioEngine(sample_rate=48000, block_size=256)
    inst = Instrument("Test", 0)
    e.instruments.append(inst)
    for i, (name, wav) in enumerate([("Aire", "air_whoosh.wav"),
                                     ("Boom", "deep_impact.wav")]):
        p = LayerParams(name=name, loop=(i == 0),
                        trigger_mode="gate" if i == 0 else "one_shot",
                        x_param="speed" if i == 0 else None,
                        y_param="gain" if i == 0 else None)
        l = Layer(p, Sample.from_file(os.path.join(DEMO, wav)), e.sample_rate)
        l.uid = e.alloc_layer_uid()
        inst.add_layer(l)
    s = e.new_surface("XY")
    s.continuous = True
    return e, inst, s


def test_produces_audio_and_xy_modulates():
    e, inst, s = build_engine()
    e.set_xy_target(s.id, 0.2, 0.5)
    a = np.concatenate([e.process_block(256) for _ in range(120)])   # ~2.5 s: deja
    e.set_xy_target(s.id, 1.0, 1.0)                                  # crecer el attack
    b = np.concatenate([e.process_block(256) for _ in range(120)])   # y estabilizar Y
    assert np.abs(a).max() > 0.01 and np.abs(b).max() > 0.01
    assert not np.allclose(a, b)          # el XY cambia realmente el sonido
    assert e.frames_total == 240 * 256    # reloj de frames correcto


def test_recording_take_has_master_stems_and_automation():
    e, inst, s = build_engine()
    e.start_recording(video_pos_s=12.5, capture_stems=True)
    for i in range(50):
        e.set_xy_target(s.id, i / 50, 0.5)
        e.process_block(256)
    take = e.stop_recording()
    assert take.start_video_s == 12.5
    assert take.frames_master.shape == (50 * 256, 2)
    assert len(take.frames_layers) == 2
    assert len(take.xy_events) == 50      # un evento por bloque, sincronizado al reloj
    # el timestamp del último evento es el inicio de su bloque: la diferencia con
    # la duración total es exactamente un bloque (sincronización muestra a muestra)
    assert abs(take.length_s - take.xy_events[-1][0] - 256 / 48000) < 1e-9


def test_render_is_deterministic():
    e, inst, s = build_engine()
    e.start_recording(0.0, True)
    for i in range(40):
        e.set_xy_target(s.id, i / 40, np.sin(i / 40))
        e.process_block(256)
    take = e.stop_recording()
    r1 = e.render_toma(take)
    r2 = e.render_toma(take)
    n = min(r1["master"].shape[0], r2["master"].shape[0])
    assert np.abs(r1["master"][:n] - r2["master"][:n]).max() == 0.0


def test_gate_mode_silence_when_not_pressed():
    e, inst, s = build_engine()
    s.continuous = False
    out = np.concatenate([e.process_block(256) for _ in range(20)])
    assert np.abs(out).max() < 1e-6       # sin presión -> silencio
    s.pressed = True
    out2 = np.concatenate([e.process_block(256) for _ in range(20)])
    assert np.abs(out2).max() > 0.01      # al pulsar suena de inmediato


def test_one_shot_trigger():
    e, inst, s = build_engine()
    s.continuous = False
    boom = inst.layers[1]
    e.trigger_layer(boom)
    out = np.concatenate([e.process_block(256) for _ in range(40)])
    assert np.abs(out).max() > 0.01


def test_clock_seconds_sync():
    e, inst, s = build_engine()
    for _ in range(96):                   # 96*256 = 24576 muestras = 0.512 s
        e.process_block(256)
    assert abs(e.clock_seconds() - 0.512) < 1e-9
