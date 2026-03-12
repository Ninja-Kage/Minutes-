import torch
import soundfile as sf
import whisperx
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import os 
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = "cpu"

def run(audio_file="meeting.wav"):
    # --- Transcribe with whisperx on CPU ---
    model = whisperx.load_model("base", device=DEVICE, compute_type="int8")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=8)

    # --- Load audio as tensor (avoids torchcodec/FFmpeg issue) ---
    waveform, sample_rate = sf.read(audio_file, dtype="float32")
    waveform = torch.tensor(waveform).unsqueeze(0)  # shape: (1, time)
    audio_dict = {"waveform": waveform, "sample_rate": sample_rate}

    # --- Diarize on CPU ---
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN
    )
    pipeline.to(torch.device("cpu"))
    raw_output = pipeline(audio_dict)

    # Support both old (Annotation) and new (DiarizeOutput) pyannote API
    if hasattr(raw_output, "speaker_diarization"):
        diarization = raw_output.speaker_diarization
    else:
        diarization = raw_output

    # --- Match speakers to transcript segments ---
    output = []
    for seg in result["segments"]:
        mid = (seg["start"] + seg["end"]) / 2
        speaker = "UNKNOWN"

        for turn, _, spk in diarization.itertracks(yield_label=True):
            if turn.start <= mid <= turn.end:
                speaker = spk
                break

        output.append({
            "speaker": speaker,
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip()
        })

    return output