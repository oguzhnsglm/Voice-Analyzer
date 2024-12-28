import torch
import torchaudio
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, pipeline, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
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

def analyze_audio_sentiment(audio_file_path):
    """
    Basit bir ses duygu analizi.
    Ses enerjisi ve sıfır geçiş oranına dayalı olarak pozitif, nötr ve negatif sonuçları döndürür.
    """
    # Ses dosyasını yükle
    speech_array, sampling_rate = torchaudio.load(audio_file_path)

    # Yeniden örnekleme (16000 Hz'e)
    if sampling_rate != 16000:
        resample = Resample(orig_freq=sampling_rate, new_freq=16000)
        speech_array = resample(speech_array)

    # RMS Enerji
    rms_energy = torch.sqrt(torch.mean(speech_array ** 2)).item()

    # Sıfır Geçiş Oranı
    zero_crossings = torch.mean((speech_array[0][:-1] * speech_array[0][1:] < 0).float()).item()

    # Dinamik skor hesaplama
    # Pozitif enerji seviyesini artırırken sıfır geçiş oranına dayalı negatiflik tahmini
    positive_score = rms_energy * 100
    neutral_score = (1 - abs(zero_crossings - 0.5)) * 50  # Dengeli sıfır geçiş oranı
    negative_score = (1 - rms_energy + zero_crossings) * 50

    # Normalize ederek yüzdelik değerlere dönüştür
    total_score = positive_score + neutral_score + negative_score
    if total_score > 0:
        sentiment_scores = {
            "positive": (positive_score / total_score) * 100,
            "neutral": (neutral_score / total_score) * 100,
            "negative": (negative_score / total_score) * 100,
        }
    else:
        sentiment_scores = {"positive": 0, "neutral": 0, "negative": 0}

    return sentiment_scores

def combined_sentiment_analysis(text, audio_file_path):
    """
    Metin ve ses dosyasının duygu analizlerini yapar ve sonuçların ortalamasını alır.
    """
    # Metin analizi
    text_scores = analyze_text_sentiment(text)
    
    # Ses analizi
    audio_scores = analyze_audio_sentiment(audio_file_path)
    
    # Ortalamalarını alarak birleşik duygu analizi oluştur
    combined_scores = {
        'positive': (text_scores['positive'] + audio_scores.get('positive', 0)) / 2,
        'neutral': (text_scores['neutral'] + audio_scores.get('neutral', 0)) / 2,
        'negative': (text_scores['negative'] + audio_scores.get('negative', 0)) / 2,
    }
    return {'text_scores': text_scores, 'audio_scores': audio_scores, 'combined_scores': combined_scores}

if __name__ == "__main__":
    sample_text = "Bugün harika hissediyorum!"  # Kullanıcı tarafından girilen metin
    audio_path = "kayit.wav"  # Ses dosyanızın yolu
    
    result = combined_sentiment_analysis(sample_text, audio_path)
    if result:
        print("Duygu Analizi Sonucu:", result) 
