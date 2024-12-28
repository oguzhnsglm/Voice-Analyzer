from transformers import pipeline
from translate_text import translate_to_english  # Çeviri fonksiyonunu import et

# Konu başlıkları
topics = ["Technology", "Health", "Finance", "Education", "Entertainment", "Science", "Politics", "Sports"]

# Zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def analyze_text_topic(text):
    """Metnin konusunu analiz eder."""
    # Türkçe metni İngilizce'ye çevirme
    translated_text = translate_to_english(text)
    if not translated_text:
        return "Çeviri yapılamadı."

    # Konu analizi
    try:
        result = classifier(translated_text, topics, multi_label=True)
        top_topic = result["labels"][0]  # En yüksek olasılıklı konu
    except Exception as e:
        print(f"Konu analizi hatası: {e}")
        return "Belirli bir konu bulunamadı."

    return top_topic

# Örnek test
if __name__ == "__main__":
    sample_text = "Yapay zeka teknolojisi hızla gelişiyor."
    topic = analyze_text_topic(sample_text)
    print("Belirlenen Konu:", topic)
