# Licencias de terceros

| Paquete | Licencia | Nota |
|---|---|---|
| numpy | BSD-3 | OK uso comercial/gratuito |
| sounddevice | MIT | Incluye binarios de PortAudio (MIT-like) |
| PortAudio | MIT-style | OK |
| soundfile | BSD | wrapper de libsndfile |
| libsndfile | LGPL-2.1 | Enlazado dinámicamente por la wheel → cumple LGPL |
| scipy | BSD-3 | OK |
| PySide6 | LGPLv3 | Permitido en apps cerradas si el enlace es dinámico y se permite sustituir la librería; incluir este archivo en el instalador |
| mido | MIT | Solo se importa si hay dispositivos MIDI |
| FFmpeg (si se usa como fallback de vídeo) | LGPL/GPL según build | Usar build LGPL (sin --enable-gpl) o evitarlo: QtMultimedia usa los códecs del SO |

Comprobación hecha el 2026-10-25. Antes de distribuir instaladores, verificar versiones concretas.
