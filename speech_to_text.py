# speech_to_text.py

import speech_recognition as sr

def transcribe_audio(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
    try:
        transcript = recognizer.recognize_google(audio, language="tr-TR")
        return transcript
    except sr.UnknownValueError:
        return None  # Sessizlik durumunda None döndür
    except sr.RequestError as e:
        return f"Could not request results; {e}"
