import whisper
import time

model = whisper.load_model("base")

def transcribe_audio(file_path):
    start_time = time.time()
    result = model.transcribe(file_path)
    end_time = time.time()

    text = result['text']
    duration = round(result['segments'][-1]['end'], 2) if result['segments'] else 0
    processing_time = round(end_time - start_time, 2)
    rtf = round(processing_time / duration, 2) if duration > 0 else 0

    return text, duration, rtf