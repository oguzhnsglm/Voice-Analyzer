from moviepy.editor import AudioFileClip
import os

# MP3 dosyasını WAV formatına dönüştürme
def convert_mp3_to_wav(input_path, output_path):
    try:
        audio = AudioFileClip(input_path)
        audio.write_audiofile(output_path, codec='pcm_s16le')
        audio.close()
        print(f"Dönüşüm başarılı: {output_path}")
    except Exception as e:
        print(f"Dosya dönüştürülürken hata oluştu: {input_path} -> {e}")

# Tüm MP3 dosyalarını belirtilen klasörde dönüştürme
def convert_all_mp3_in_folder(input_folder, output_folder):
    for subfolder in ['m', 'w']:  # 'm' ve 'w' klasörlerini ayrı işliyoruz
        input_subfolder = os.path.join(input_folder, subfolder)
        output_subfolder = os.path.join(output_folder, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)

        for file in os.listdir(input_subfolder):
            if file.endswith('.mp3'):
                input_path = os.path.join(input_subfolder, file)
                output_path = os.path.join(output_subfolder, os.path.splitext(file)[0] + '.wav')
                convert_mp3_to_wav(input_path, output_path)

# Kullanım
if __name__ == "__main__":
    input_folder = 'data'
    output_folder = 'converted_wav'
    convert_all_mp3_in_folder(input_folder, output_folder)
