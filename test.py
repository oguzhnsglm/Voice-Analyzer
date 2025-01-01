'''import sys
import os

# Kök dizini sys.path'e ekleyin
sys.path.append(os.path.abspath(os.path.dirname(__file__)))'''
import unittest
from speech_to_text import transcribe_audio
from sentiment_analysis import analyze_text_sentiment  # Burada sentiment_analysis_module, kodun yer aldığı dosya adıdır.
from topic_analysis import analyze_text_topic
class Test(unittest.TestCase):
    def test_transcribe_audio_success(self):
        # Doğru bir ses dosyası kullanarak testi çalıştırın
        audio_file_path = "tests/sample_audio.wav"  # Test ses dosyasının yolu
        expected_transcript = "Merhaba dünya"  # Beklenen metin
        transcript = transcribe_audio(audio_file_path)
        self.assertEqual(transcript, expected_transcript, "Transkript eşleşmiyor!")

    def test_transcribe_audio_word_count(self):
        expected_word_count = 2  # Beklenen kelime sayısı: "Merhaba dünya"
        audio_file_path = "tests/sample_audio.wav"  # Test ses dosyasının yolu
        transcript = transcribe_audio(audio_file_path) 
        word_count = len(transcript.split())
        self.assertEqual(word_count, expected_word_count, "Kelime sayısı eşleşmiyor!")

    def test_positive_sentiment(self):
        """Pozitif bir metni analiz eder."""
        text = "Harika bir gün geçirdim."
        result = analyze_text_sentiment(text)

        self.assertIn("positive", result, "Pozitif duygu sonucu bulunamadı!")
        self.assertIn("neutral", result, "Nötr duygu sonucu bulunamadı!")
        self.assertIn("negative", result, "Negatif duygu sonucu bulunamadı!")
        
        # Pozitif duygu skoru en yüksek olmalı
        self.assertGreater(result["positive"], result["neutral"], "Pozitif skor, nötr skordan büyük olmalı!")
        self.assertGreater(result["positive"], result["negative"], "Pozitif skor, negatif skordan büyük olmalı!")

    def test_negative_sentiment(self):
        """Negatif bir metni analiz eder."""
        text = "Konser çok kötüydü."
        result = analyze_text_sentiment(text)

        self.assertIn("positive", result, "Pozitif duygu sonucu bulunamadı!")
        self.assertIn("neutral", result, "Nötr duygu sonucu bulunamadı!")
        self.assertIn("negative", result, "Negatif duygu sonucu bulunamadı!")

        # Negatif duygu skoru en yüksek olmalı
        self.assertGreater(result["negative"], result["neutral"], "Negatif skor, nötr skordan büyük olmalı!")
        self.assertGreater(result["negative"], result["positive"], "Negatif skor, pozitif skordan büyük olmalı!")

    def test_analyze_text_topic_success(self):
        """
        Başarılı bir konu analizi testi.
        """
        sample_text = "Yapay zeka teknolojisi hızla gelişiyor."  # Test metni
        expected_topic = "Technology"  # Beklenen konu
        result = analyze_text_topic(sample_text)
        
        self.assertEqual(result, expected_topic, f"Konu analizi hatalı! Beklenen: {expected_topic}, Alınan: {result}")



if __name__ == "__main__":
    unittest.main()
