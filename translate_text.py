from transformers import pipeline

# Çeviri pipeline
translator = pipeline(task="translation", model="Helsinki-NLP/opus-mt-tr-en")

def translate_to_english(text):
    """Türkçe metni İngilizce'ye çevirir."""
    try:
        translated_text = translator(text, max_length=512)[0]["translation_text"]
        return translated_text
    except Exception as e:
        print(f"Çeviri hatası: {e}")
        return None
