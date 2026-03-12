import sounddevice as sd
import numpy as np
import wave

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

audio_chunks = []
stream = None

def start():
    global audio_chunks, stream
    audio_chunks = []
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, callback=_callback)
    stream.start()

def _callback(indata, frames, time, status):
    audio_chunks.append(indata.copy())

def stop():
    global stream
    if stream:
        stream.stop()
        stream.close()
        stream = None

def save(filename="meeting.wav"):
    if not audio_chunks:
        return None
    audio = np.concatenate(audio_chunks, axis=0)
    with wave.open(filename, "w") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return filename