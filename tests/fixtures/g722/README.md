# G.722 cross-check fixtures

Used by `tests/g722.rs` to check the encoder and decoder bit-exact against an
independent implementation.

- `input.s16le`: 0.5 s of 16 kHz mono s16le. A 100 Hz to 7.5 kHz chirp at
  half scale with a little white noise (numpy, seed 7), and a burst near full
  scale at samples 6000 to 6400. It covers both sub-bands and clipping.
- `ffmpeg.g722`: `input.s16le` encoded by FFmpeg 9.0.2's G.722 encoder
  (`ffmpeg -f s16le -ar 16000 -ac 1 -i input.s16le -c:a g722 -f g722 ffmpeg.g722`,
  default options, so no trellis search).
- `ffmpeg_decoded.s16le`: `ffmpeg.g722` decoded by FFmpeg's G.722 decoder
  (`ffmpeg -f g722 -i ffmpeg.g722 -f s16le -ar 16000 -ac 1 ffmpeg_decoded.s16le`).
