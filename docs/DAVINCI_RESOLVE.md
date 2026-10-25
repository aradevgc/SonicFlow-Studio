# Guía: SonicFlow Studio + DaVinci Resolve / Fairlight

Resolve no admite plugins VST/AU externos arbitrarios para apps independientes, por lo que la
integración fiable es **por archivos WAV con timecode compartido**:

1. Importa tu clip en SonicFlow Studio y colócalo en el punto donde empieza la acción a sonorizar.
2. Pulsa REC: la toma guarda la posición exacta del vídeo en su inicio (`start_video_s`).
3. Interpreta el efecto con la superficie XY mientras ves el vídeo.
4. Exporta la toma: mezcla estéreo + stems por capa en WAV 24-bit/48 kHz. El nombre del archivo
   incluye el offset de inicio (p. ej. `Toma1@00-01-23.500.wav`) y se genera un `.csv` de
   metadatos (TC de inicio, duración, fps del vídeo).
5. En Resolve: importa el vídeo original en la misma pista temporal; arrastra cada WAV a una pista
   de audio nueva y muévelo hasta el timecode indicado (activa "Snap" y el modo de ripple off).
   Al tener ambos la misma tasa (48 kHz) y origen, la alineación es muestra-exacta.
6. Para re-grabar: crea Toma 2 sin borrar Toma 1 (el sistema conserva todas las tomas) y exporta
   solo la que te guste.

Nota: trabaja con el proyecto de Resolve a 48 kHz para evitar re-muestreos. Si tu timeline es 24/25/30 fps,
el CSV incluye el frame exacto de inicio calculado con el fps del vídeo.
