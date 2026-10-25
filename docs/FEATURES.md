# Funciones implementadas vs pendientes

Actualizado: 2026-10-25

## ✅ Implementado y PROBADO (ver `scripts/demo_engine.py`, `tests/`)

Motor de audio (`sonicflow/audio/`):
- Thread de audio de baja latencia con PortAudio (sounddevice); backend "dummy" con reloj interno
  si no hay dispositivo físico (la app sigue pudiendo grabar/exportar).
- Instrumentos con múltiples capas; por capa: sample, gain dB, pan, pitch, speed, filtro LP/HP/BP,
  drive (saturación tanh), envíos a delay y reverb maestros, ADSR sin clics, loop, modo gate/one-shot.
- Superficies XY múltiples: objetivo escrito desde UI/MIDI, suavizado exponencial por bloque,
  sensibilidad, deadzone, curvas de respuesta, zonas de activación, inversión de ejes, retorno al
  centro, modo continuo/pulsado, asignación X/Y a cualquier parámetro por capa con rangos.
- Lectura de muestra con fase fraccional continua + interpolación lineal (sin clics ni saltos).
- Grabación DENTRO del process_block: master + stems por capa + automation XY (estado suavizado por
  bloque) + posición de inicio del vídeo (timecode).
- Replay de automation grabada sobre el motor en vivo; undo/redo de tomas.
- Render offline determinista (`render_toma`) con el MISMO DSP: dos renders consecutivos dan
  diferencia máxima 0.0 (probado); exportación WAV 24-bit y stems verificada.
- Carga de muestras WAV/FLAC/OGG/MP3/AIFF con re-muestreo polyphase Kaiser (~140 dB) y fallback
  stdlib cuando falte soundfile/scipy.
- Medición: peak/RMS maestro y por capa (para medidores de la UI).

## ⏳ Pendiente (código de soporte preparado, sin UI final)

- `main.py` + interfaz PySide6 (ventana principal, panel de vídeo, superficie XY dibujada,
  mezclador con medidores, biblioteca, ajustes de audio).
- Reproductor de vídeo (QtMultimedia): importar MP4/MOV, transporte, frame-step, bucle A-B,
  velocidad, ventana separada, envío de posición al motor para sincronizar grabaciones.
- Biblioteca de sonidos (carpetas, categorías, etiquetas, favoritos, preescucha).
- MIDI (mido): mapeo de faders/knobs/pads/joystick XY + Learn; botones de transporte.
- Guardado/carga de proyectos y presets (JSON atómico ya disponible en `utils.py`).
- Exportación multi-toma con nombres + metadatos TC para Resolve (la base ya existe en el motor).
- Instaladores (.dmg / .exe vía PyInstaller) y empaquetado final.

Ninguna función listada como pendiente debe considerarse terminada: no está probada end-to-end.
