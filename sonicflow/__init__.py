"""
SonicFlow Studio — Alternativa gratuita y open source inspirada en el flujo de
trabajo de Krotos Studio: diseño e interpretación de efectos de sonido en tiempo
real mientras se reproduce un vídeo.

Arquitectura (separación estricta de capas):

    sonicflow/
      audio/       Motor de audio DSP en tiempo real (thread de audio, sin UI)
      video/       Reproductor de vídeo (PySide6 QtMultimedia / ffmpeg de respaldo)
      library/     Biblioteca local de sonidos (JSON + categorías + etiquetas)
      midi/        Control MIDI y aprendizaje (mido)
      project/     Persistencia de proyectos, presets y tomas
      ui/          Interfaz (PySide6): superficie XY, mezclador, transporte...
      utils.py     Utilidades comunes (timecode, atomic JSON, logging)

Licencias de dependencias: ver THIRD_PARTY_LICENSES.md
"""

__version__ = "1.0.0"
APP_NAME = "SonicFlow Studio"
