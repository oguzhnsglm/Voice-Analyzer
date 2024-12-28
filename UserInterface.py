import os
import tkinter as tk
from tkinter import font, messagebox
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from pydub import AudioSegment
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
import librosa
from tensorflow.keras.models import load_model
from tkinter import messagebox
import time
import threading
from topic_analysis import analyze_text_topic
from speech_to_text import transcribe_audio
from AudioMl4 import process_data_and_train, predict_user
from sentiment_analysis import combined_sentiment_analysis


class AudioRecorder:
    def __init__(self, root):
        self.root = root
        self.root.title("Speaker Recognition and Sentiment Analysis")
        self.is_recording = False
        self.frames = []

        entry_frame = tk.Frame(root)
        entry_frame.pack(padx=10, pady=10)

        # Kullanıcı adı girişi
        tk.Label(entry_frame, text="Kullanıcı Adı:").pack(side=tk.LEFT)
        self.name_entry = tk.Entry(entry_frame, width=30)
        self.name_entry.pack(side=tk.LEFT)


        button_frame = tk.Frame(root)
        button_frame.pack(padx=10, pady=10)
        self.model_trained = False  # Model eğitilmediği için başlangıç değeri False

        button_style = {
            "font": ("Arial", 12, "normal"),
            "bg": "#4CAF50",
            "fg": "#FFFFFF",
            "activebackground": "#45a049",
            "activeforeground": "#FFFFFF",
            "relief": "raised",
            "bd": 3,
            "width": 15
        }

        self.train_button = tk.Button(button_frame, text="Model Eğit", command=self.train_model, state=tk.DISABLED, **button_style)
        self.train_button.pack(side=tk.LEFT, padx=5, pady=10)

        self.start_button = tk.Button(button_frame, text="Kayıt Başlat", command=self.start_recording, **button_style)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=10)

        self.stop_button = tk.Button(button_frame, text="Kayıt Durdur", command=self.stop_recording, state=tk.DISABLED, **button_style)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=10)

        self.save_button = tk.Button(button_frame, text="Kayıt Kaydet", command=self.save_recording, state=tk.DISABLED, **button_style)
        self.save_button.pack(side=tk.LEFT, padx=5, pady=10)

        self.process_button = tk.Button(button_frame, text="Kayıdı İşle", command=self.process_recording, state=tk.DISABLED, **button_style)
        self.process_button.pack(side=tk.LEFT, padx=5, pady=10)

        self.signal_plot = plt.figure(figsize=(6, 3))
        self.ax = self.signal_plot.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.signal_plot, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.X)

        self.histogram_plot = plt.figure(figsize=(6, 2))
        self.hist_ax = self.histogram_plot.add_subplot(111)
        self.hist_canvas = FigureCanvasTkAgg(self.histogram_plot, master=root)
        self.hist_canvas.get_tk_widget().pack(fill=tk.X)

        self.info_text = tk.Text(root, height=10, wrap=tk.WORD)
        self.info_text.pack(fill=tk.X)

        self.info_text.tag_configure("header", font=("Arial", 12, "bold"), foreground="#4CAF50")
        self.info_text.tag_configure("content", font=("Arial", 12, "normal"), foreground="#000000")

        self.model_path = 'speaker_recognition_model.h5'
        self.label_encoder_path = 'label_encoder.pkl'
        self.model = None
        self.label_encoder = None

        self.saniye_basina_ornek = 44100
        self.saniye = 5
        
    def lock_buttons(self):
        for button in [self.train_button, self.start_button, self.stop_button, self.save_button, self.process_button]:
            button.config(state=tk.DISABLED)

    def unlock_buttons(self):
        self.start_button.config(state=tk.NORMAL if not self.is_recording and self.model_trained else tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.process_button.config(state=tk.DISABLED)
        if not self.model_trained:
            self.train_button.config(state=tk.NORMAL if self.save_button["state"] == tk.NORMAL else tk.DISABLED)
        else:
            self.train_button.config(state=tk.DISABLED)

    def train_model(self):
        def train_and_check():
            self.lock_buttons()
            self.train_button.config(text="Model Eğitiliyor...")
            folder_path = 'converted_wav'
            csv_path = 'audio_features.csv'

            process_data_and_train(folder_path, csv_path)

            while not os.path.exists(self.model_path):
                time.sleep(1)

            self.model = load_model(self.model_path)
            self.label_encoder = joblib.load(self.label_encoder_path)

            self.model_trained = True
            self.train_button.config(text="Model Eğitildi", state=tk.DISABLED)
            self.unlock_buttons()
            messagebox.showinfo("Başarılı", "Model başarıyla eğitildi ve yüklendi!")

        threading.Thread(target=train_and_check).start()

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.frames = []
            self.stream = sd.InputStream(callback=self.callback, channels=1, samplerate=self.saniye_basina_ornek)
            self.stream.start()
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.train_button.config(state=tk.DISABLED)
            self.root.after(100, self.update_ui)

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.stream.stop()
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.DISABLED)
            self.save_button.config(state=tk.NORMAL)
            # Model mevcutsa "Kayıdı İşle" butonunu aktif et
            if os.path.exists(self.model_path):
                self.process_button.config(state=tk.NORMAL)
            else:
                self.process_button.config(state=tk.DISABLED)
            self.plot_signal()
            self.plot_histogram()
             # Kayıt işlemini gerçekleştir
            fs = self.saniye_basina_ornek
            audio_data = np.concatenate(self.frames, axis=0)
            wav.write("kayit.wav", fs, audio_data)

            sound = AudioSegment.from_wav("kayit.wav")
            sound.export("kayit1_pcm.wav", format="wav")
            

    def save_to_data_folder(self, user_name):
        """
        Kaydı belirli bir klasöre kaydeder.
        
        Parameters:
        - user_name (str): Kullanıcının belirttiği isim.
        """
        data_folder = "converted_wav"
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        # Dosya adını kullanıcı adı ve zaman bilgisiyle oluştur
        timestamp = int(time.time())
        file_name = f"{user_name}_{timestamp}.wav"
        destination_file = os.path.join(data_folder, file_name)

        source_file = "kayit1_pcm.wav"
        if os.path.exists(source_file):
            os.rename(source_file, destination_file)
            return file_name
        else:
            raise FileNotFoundError("Geçici ses dosyası bulunamadı. Kaydetme işlemi başarısız oldu.")
    

    def callback(self, indata, frames, time, status):
        self.frames.append(indata.copy())

    def update_ui(self):
        if self.is_recording:
            try:
                self.plot_signal()
                self.plot_histogram()
                self.root.after(100, self.update_ui)
            except Exception as e:
                print(f"UI güncellenirken hata: {e}")


    def save_recording(self):
        """
        Kaydı kaydetmek için gerekli işlemleri yapar.
        """
        if self.frames:
            try:
                # Kullanıcı adını kontrol et
                user_name = self.name_entry.get().strip()  # Kullanıcı girişinden adı al
                if not user_name:
                    messagebox.showerror("Hata", "Lütfen bir isim giriniz!")
                    return

                # Kaydı veri klasörüne taşı
                file_name = self.save_to_data_folder(user_name)

                # TextBox'u temizle
                self.name_entry.delete(0, tk.END)  # TextBox içeriğini siler


                # Butonları aktif et
                self.train_button.config(state=tk.NORMAL)  # Model eğit butonu aktif edilir
                self.start_button.config(state=tk.NORMAL)  # Kayıt başlat butonu aktif edilir
                self.save_button.config(state=tk.DISABLED)  # Kayıt kaydet butonu devre dışı bırakılır
                self.process_button.config(state=tk.DISABLED)  # İşleme butonu devre dışı bırakılır

                # Kullanıcıya bilgi mesajı göster
                messagebox.showinfo("Başarılı", f"Kayıt '{file_name}' olarak kaydedildi ve veri klasörüne taşındı!")
            except Exception as e:
                messagebox.showerror("Hata", f"Kayıt kaydedilirken hata oluştu: {e}")



    def reset_application(self):
        """
        Uygulamayı ilk açıldığındaki duruma döndürür.
        """
        self.is_recording = False
        self.frames = []  # Kayıt verilerini sıfırla
        self.lock_buttons()  # Tüm düğmeleri kilitle
        self.start_button.config(state=tk.NORMAL)  # Sadece Kayıt Başlat aktif olur
        self.train_button.config(state=tk.DISABLED)  # Model Eğit butonunu pasif yap
        self.save_button.config(state=tk.DISABLED)  # Kayıt Kaydet butonunu pasif yap
        self.process_button.config(state=tk.DISABLED)  # Kayıdı İşle butonunu pasif yap
        self.stop_button.config(state=tk.DISABLED)  # Kayıt Durdur butonunu pasif yap
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)  # Bilgi ekranını temizle
        self.info_text.config(state=tk.DISABLED)

    def process_recording(self):
        try:
            file = "kayit1_pcm.wav"
            if not os.path.exists(file):
                messagebox.showerror("Hata", "İşlenecek kayıt dosyası bulunamadı!")
                return
            kelime=transcribe_audio(file)
            if kelime is None:
                messagebox.showerror("Hata", "Kayıt işlenirken hata oluştu!")
                self.reset_application()
            else:    
                # Kayıt dosyasını işleyin
                transcript, kelime_sayisi = self.getWords(file)
                tahmin = self.speaker_identification(file)
                topic = analyze_text_topic(transcript)

                # Duygu analizini gerçekleştirin
                text_sentiment, audio_sentiment, combined_sentiment = self.perform_sentiment_analysis(transcript, file)

                # Sonuçları güncelleyin
                self.update_info_text(tahmin, transcript, kelime_sayisi, topic, text_sentiment, audio_sentiment, combined_sentiment)

                # Kullanıcıya bilgi mesajı
                messagebox.showinfo("Başarılı", "Kayıt başarıyla işlendi!")

                # Kayıt başlat butonunu yeniden aktif hale getirin
                self.lock_buttons()
                self.start_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Hata", f"Kayıt işlenirken hata oluştu: {e}")


    def getWords(self, file):
        transcript = transcribe_audio(file)
        kelime_sayisi = len(transcript.split())
        return transcript, kelime_sayisi

    def speaker_identification(self, file):
        y, sr = librosa.load(file, sr=self.saniye_basina_ornek)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
        mfcc = np.mean(mfcc.T, axis=0)

        model = load_model(self.model_path)
        label_encoder = joblib.load(self.label_encoder_path)

        prediction = model.predict(mfcc.reshape(1, -1))
        tahmin_indeksi = np.argmax(prediction, axis=1)
        tahmin_isim = label_encoder.inverse_transform(tahmin_indeksi)

        return tahmin_isim[0]

    def perform_sentiment_analysis(self, text, audio_file_path):
        sentiment_results = combined_sentiment_analysis(text, audio_file_path)
        return sentiment_results['text_scores'], sentiment_results['audio_scores'], sentiment_results['combined_scores']


    def update_info_text(self, tahmin, transcript, kelime_sayisi, topic, text_sentiment, audio_sentiment, combined_sentiment):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)

        self.info_text.insert(tk.END, "Konuşmacı: ", "header")
        self.info_text.insert(tk.END, f"{tahmin}\n", "content")

        self.info_text.insert(tk.END, "Transcript: ", "header")
        self.info_text.insert(tk.END, f"{transcript}\n", "content")

        self.info_text.insert(tk.END, "Kelime Sayısı: ", "header")
        self.info_text.insert(tk.END, f"{kelime_sayisi}\n", "content")

        self.info_text.insert(tk.END, "Belirlenen Konu: ", "header")
        self.info_text.insert(tk.END, f"{topic}\n", "content")

        self.info_text.insert(tk.END, "Metin Duygu Analizi: ", "header")
        self.info_text.insert(tk.END, f"{text_sentiment}\n", "content")

        self.info_text.insert(tk.END, "Ses Duygu Analizi: ", "header")
        self.info_text.insert(tk.END, f"{audio_sentiment}\n", "content")

        self.info_text.insert(tk.END, "Birleştirilmiş Duygu Analizi: ", "header")
        self.info_text.insert(tk.END, f"{combined_sentiment}\n", "content")

        self.info_text.config(state=tk.DISABLED)


    def plot_histogram(self):
        if self.frames:
            audio_data = np.concatenate(self.frames, axis=0)
            self.hist_ax.clear()
            self.hist_ax.hist(audio_data, bins=100, color='b', alpha=0.7)
            self.hist_ax.set_title('Ses Verisi Dağılımı')
            self.hist_ax.set_xlabel('Amplitüd')
            self.hist_ax.set_ylabel('Frekans')
            self.hist_canvas.draw()

    def plot_signal(self):
        if self.frames:
            audio_data = np.concatenate(self.frames, axis=0)
            self.ax.clear()
            self.ax.plot(np.linspace(0, len(audio_data) / self.saniye_basina_ornek, num=len(audio_data)), audio_data)
            self.ax.set_title('Ses Sinyali')
            self.ax.set_xlabel('Zaman (s)')
            self.ax.set_ylabel('Amplitüd')
            self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x700")
    app = AudioRecorder(root)
    root.mainloop()
