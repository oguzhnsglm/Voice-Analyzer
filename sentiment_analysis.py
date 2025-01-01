import torch
import torchaudio
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, pipeline, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, AutoProcessor, AutoModelForAudioClassification
from torchaudio.transforms import Resample
from scipy.special import softmax
from translate_text import translate_to_english  # Çeviri fonksiyonunu import et

processor = Wav2Vec2Processor.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
model = Wav2Vec2ForSequenceClassification.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")

# Cardiff NLP modeli
text_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(text_model_name)
config = AutoConfig.from_pretrained(text_model_name)
text_model = AutoModelForSequenceClassification.from_pretrained(text_model_name)

def analyze_text_sentiment(text):
    """
    Cardiff NLP modeline dayalı metin duygu analizi.
    Pozitif, negatif ve nötr sonuçları yüzdelik olarak döndürür.
    """
    translated_text = translate_to_english(text)
    # Metni tokenize edip modelden geçirin
    encoded_input = tokenizer(translated_text, return_tensors='pt')
    output = text_model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

       # Sıralama işlemi
    ranking = np.argsort(scores)[::-1]

    # Sıralı sonuçları sentiment_scores formatına dönüştürme
    sentiment_scores = {}
    for i in range(scores.shape[0]):
        label = config.id2label[ranking[i]]
        sentiment_scores[label.lower()] = scores[ranking[i]]*100
        print(f"{i+1}) {label} {scores[ranking[i]]*100:.2f}%")

    return sentiment_scores

# AutoProcessor ve model ile manuel duygu analizi
def analyze_audio_sentiment(audio_path):
    """AutoProcessor kullanarak ses dosyasının duygu analizi."""
    try:
        # Ses dosyasını yükleme
        waveform, sample_rate = torchaudio.load(audio_path)

        # Yeniden örnekleme (16000 Hz)
        if sample_rate != 16000:
            resampler = Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # İşleme uygun hale getirme
        inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")

        # Modelden tahmin
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()[0]

        # Sonuçları etiketlerle eşleştirme
        labels = model.config.id2label
        results = {labels[i]: scores[i] for i in range(len(labels))}
        print(results)
        return results

    except Exception as e:
        print(f"Ses duygu analizi sırasında hata oluştu: {e}")
        return {"positive": 0, "neutral": 0, "negative": 0}


def combined_sentiment_analysis(text, audio_file_path):
    """
    Metin ve ses dosyasının duygu analizlerini yapar ve sonuçların ortalamasını alır.
    Ses analizindeki `arousal`, `dominance`, `valence` değerlerini, 
    metin analizindeki `positive`, `neutral` ve `negative` duygularla eşleştirir.
    
    Returns:
        dict: {'text_scores': dict, 'audio_scores': dict, 'combined_scores': dict}
    """
    # Metin analizi
    text_scores = analyze_text_sentiment(text)

    # Ses analizi
    audio_scores = analyze_audio_sentiment(audio_file_path)

    # Ses analizindeki değerleri normalize ederek yüzdelik hale getir
    positive_audio = audio_scores['valence'] * 100
    neutral_audio = audio_scores['dominance'] * 100
    negative_audio = audio_scores['arousal'] * 100

    # Ses analizini metin analizine bağlamak için ağırlıklar belirle
    audio_weight = 0.4  # Ses analizinin ağırlığı
    text_weight = 0.6   # Metin analizinin ağırlığı

    # Birleşik skorları hesapla
    combined_scores = {
        'positive': (positive_audio * audio_weight) + (text_scores['positive'] * text_weight),
        'neutral': (neutral_audio * audio_weight) + (text_scores['neutral'] * text_weight),
        'negative': (negative_audio * audio_weight) + (text_scores['negative'] * text_weight),
    }

    return {
        'text_scores': text_scores,
        'audio_scores': {
            'arousal': audio_scores['arousal'],
            'dominance': audio_scores['dominance'],
            'valence': audio_scores['valence'],
            'positive': positive_audio,
            'neutral': neutral_audio,
            'negative': negative_audio
        },
        'combined_scores': combined_scores
    }

if __name__ == "__main__":
    sample_text = "Bugün harika hissediyorum!"  # Kullanıcı tarafından girilen metin
    audio_path = "kayit.wav"  # Ses dosyanızın yolu

    result = combined_sentiment_analysis(sample_text, audio_path)
    if result:
        print("Duygu Analizi Sonucu:", result)
