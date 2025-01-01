import os
import numpy as np
import pandas as pd
import librosa
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

def extract_features(folder_path, output_csv, num_features=20):
    """
    Özellikleri çıkarır, seçer ve sonuçları CSV dosyasına kaydeder.

    Args:
        folder_path (str): Ses dosyalarının bulunduğu klasör.
        output_csv (str): Özelliklerin kaydedileceği CSV dosyası.
        num_features (int): Seçilecek en önemli özellik sayısı (default: 10).
    """
    # Tüm ses dosyalarından özellikleri çıkar
    new_data = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                try:
                    audio, sr = librosa.load(file_path, sr=None)
                    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=50)  # 50 MFCC çıkarımı
                    mfccs_mean = np.mean(mfccs.T, axis=0)
                    user_full_name = os.path.splitext(file)[0]
                    user = user_full_name.split('_')[0]  # Kullanıcı adını dosya adından ayıkla
                    row = list(mfccs_mean) + [user]
                    new_data.append(row)
                except Exception as e:
                    print(f"Hata oluştu: {file_path} -> {e}")

    # DataFrame oluştur
    columns = [f'mfcc_{i+1}' for i in range(50)] + ['user']
    df = pd.DataFrame(new_data, columns=columns)

    if df.empty:
        print("Hiçbir veri işlenmedi. Çıkılıyor...")
        return

    # Özellik ve etiketleri ayır
    X = df.iloc[:, :-1].values  # Özellikler
    y = df['user'].values       # Etiketler

    # Label encoding for target variable (y)
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # SelectKBest ile en iyi özellikleri seç (ANOVA F-Test)
    try:
        selector = SelectKBest(score_func=f_classif, k=num_features)
        X_new = selector.fit_transform(X, y_encoded)

        # Seçilen sütunları al
        selected_indices = selector.get_support(indices=True)
        selected_columns = [columns[i] for i in selected_indices] + ['user']

        # Seçilen özelliklere göre DataFrame oluştur
        reduced_df = pd.DataFrame(X_new, columns=[columns[i] for i in selected_indices])
        reduced_df['user'] = y

        # SelectKBest nesnesini kaydet
        joblib.dump(selector, 'feature_selector.pkl')
        print("SelectKBest nesnesi başarıyla kaydedildi.")

        # Sonucu CSV'ye kaydet
        reduced_df.to_csv(output_csv, index=False)
        print(f"Seçilen {num_features} özellik başarıyla {output_csv} dosyasına kaydedildi.")
    except Exception as e:
        print(f"Özellik seçimi sırasında hata oluştu: {e}")

if __name__ == "__main__":
    folder_path = 'converted_wav'
    output_csv = 'audio_features.csv'

    # Özellik çıkarımı ve seçim işlemini tamamla
    extract_features(folder_path, output_csv, num_features=10)
