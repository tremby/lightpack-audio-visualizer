# Lightpack audio visualizer

This is some old code which makes Lightpack lights react to music in real time.

- Audio data is read from a file (I used a FIFO provided by MPD)
- It is assumed to be stereo PCM at 16-bit depth and 44.1kHz sample rate
- A single Lightpack (10 LED strips) is assumed

The stereo channels are processed separately.
Fourier transforms are applied,
and over a short window of time amplitudes are collected for five frequency buckets.
Each frequency bucket for each channel has an LED strip assigned
(low frequencies at the bottom of the screen, high at the top,
left on the left, right on the right)
and light brightness is assigned based on the audio amplitude.

Hues are assigned and gradually cycled through,
to give a sort of rainbow effect.
When a gap in music is detected
a new sector of hues and hue cycle speed is chosen.

### Poor code

There's likely a lot of bad code in here
since I never took the time to really understand what the FFTs are doing.

There are a bunch of magic numbers based on observed outputs of the FFTs
in order to get a pleasing range of visual output.

### Abandoned

I no longer run Prismatik and so my Lightpack no longer provides the API this script uses.
This is here for archival purposes and in case someone else wants to play with it.
