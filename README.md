# SonicFlow Studio

Alternativa **gratuita y de código abierto** inspirada en el flujo de trabajo de Krotos Studio:
importa un vídeo, reprodúcelo e **interpreta efectos de sonido en tiempo real** con la superficie XY
(ratón / trackpad / MIDI), graba la interpretación y expórtala en WAV 24-bit sincronizada con el vídeo
(listo para DaVinci Resolve / Fairlight).

## Estructura del proyecto

```
sonicflow/
  audio/       Motor DSP en tiempo real (PortAudio vía sounddevice; thread de audio aislado de la GUI)
    engine.py    AudioEngine: instrumentos, capas, superficies XY, grabación, render offline
    dsp.py       Biquad, ADSR sin clics, delay estéreo, reverberación FDN, saturación tanh
    sample.py    Carga de muestras + re-muestreo polyphase de alta calidad + waveforms
  video/       Reproductor de vídeo (PySide6 QtMultimedia)          [pendiente]
  library/     Biblioteca local de sonidos (categorías, etiquetas)  [pendiente]
  midi/        Control MIDI + learn (mido)                          [pendiente]
  project/     Guardado/carga de proyectos (JSON atómico)           [pendiente]
  ui/          Interfaz PySide6 (superficie XY, mezclador…)         [pendiente]
  utils.py     Timecode SMPTE, escritura atómica, logging
resources/demosounds/  5 sonidos originales de demo (WAV 48 kHz, generados para el proyecto)
scripts/demo_engine.py PRUEBA FUNCIONAL del motor (grabación + exportación reales, sin GUI)
examples/              demo_take.wav (24-bit) + stems por capa ya generados
tests/                 pytest del motor y de sincronización
docs/                  guía Resolve, instalación, limitaciones, funciones
main.py                Punto de entrada de la aplicación            [pendiente]
```

## Requisitos

- Python 3.10+ (recomendado 3.12); macOS, Windows o Linux
- Sin conexión a Internet tras instalar. Sin cuentas, sin pagos, sin subidas a servidores.

## Instalación paso a paso

### macOS

```bash
# 1. Instala Python 3.12 si no lo tienes (https://python.org/downloads o `brew install python`)
# 2. Entra en la carpeta del proyecto y crea un entorno virtual:
cd SonicFlow-Studio
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*(PortAudio viene incluido en la wheel de `sounddevice`; no hace falta `brew install portaudio`.)*

### Windows

```bat
:: 1. Instala Python 3.12 desde python.org (marca "Add python.exe to PATH")
cd SonicFlow-Studio
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Comprobación rápida (sin GUI)

```bash
python scripts/demo_engine.py
```

Debe imprimir el backend de audio, grabar una toma con los movimientos XY y escribir
`examples/demo_take.wav` (estéreo 24-bit/48 kHz) más los stems por capa en `examples/demo_take_stems/`.
Con altavoces conectados, escucha la interpretación en tiempo real con:

```bash
python scripts/demo_engine.py --play
```

### Pruebas automatizadas

```bash
pytest tests/ -v
```

### Ejecutar la aplicación completa (interfaz)

```bash
python main.py
```

> **Estado actual:** el motor de audio, la superficie XY, la grabación de interpretaciones,
> el replay de automation y la exportación WAV/stems **funcionan y están probados**.
> La interfaz gráfica, el reproductor de vídeo, la biblioteca, MIDI y el guardado de proyectos
> están **pendientes**: lista exacta en [`docs/FEATURES.md`](docs/FEATURES.md).

## Asignaciones XY

Parámetros asignables por eje y por capa: `speed`, `pitch`, `filter`, `gain`, `pan`, `drive`,
`rev`, `delay`; con inversión de eje, rango personalizado, curvas de respuesta
(lineal/exp/log/sqrt/square/s-curve), zona de activación, sensibilidad y suavizado en ms.
Modo gate (pulsación mantenida), one-shot y continuo; retorno automático al centro opcional.

## Uso con DaVinci Resolve

Ver [`docs/DAVINCI_RESOLVE.md`](docs/DAVINCI_RESOLVE.md). Resumen: exporta la toma (y sus stems)
en WAV 24-bit; cada toma recuerda el tiempo de inicio del vídeo, así que en Fairlight se alinean
en el mismo timecode.

## Generar instaladores (opcional)

```bash
pip install pyinstaller
pyinstaller --windowed --name SonicFlowStudio main.py   # cuando main.py esté disponible
```

## Licencias

- Código: MIT (`LICENSE`). Dependencias: numpy (BSD), sounddevice/soundfile (MIT), libsndfile (LGPL),
  scipy (BSD), PySide6 (LGPLv3), mido (MIT) — detalles en `docs/THIRD_PARTY_LICENSES.md`.
- Los sonidos de `resources/demosounds/` son sintéticos, generados para este proyecto y libres de uso.
