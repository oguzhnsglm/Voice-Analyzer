import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import librosa
from feature_extraction import extract_features  # feature_extraction.py dosyasındaki fonksiyon

def create_model(input_shape, num_classes):
    """
    Modeli oluşturur ve derler.
    """
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def process_data_and_train(folder_path, csv_path):
    """
    Özellik çıkarımı yapar ve modeli eğitir.
    """
    # CSV dosyasını sıfırla veya oluştur
    if os.path.exists(csv_path):
        print(f"{csv_path} dosyası mevcut. İçerik sıfırlanıyor...")
        with open(csv_path, 'w') as file:
            file.write('')  # Dosya içeriğini tamamen siler
    else:
        print(f"{csv_path} dosyası oluşturulacak.")

    extract_features(folder_path, csv_path)

    # CSV dosyasından verileri al
    data = pd.read_csv(csv_path)
    X = data.iloc[:, :-1].values
    y = data['user'].values

    # Etiketleri dönüştür ve modele uygun hale getir
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    # Modeli oluştur ve eğit
    model = create_model(input_shape=X.shape[1], num_classes=y_categorical.shape[1])
    model.fit(X, y_categorical, epochs=30, batch_size=32, verbose=1)

    # Model ve etiketleri kaydet
    model.save('speaker_recognition_model.h5')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("Model ve etiketler başarıyla kaydedildi.")

def predict_user(file_path, model_path, label_encoder_path):
    """
    Verilen ses dosyasını tahmin eder.
    """
    # Özellik çıkarımı
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=10)
    features = np.mean(mfccs.T, axis=0).reshape(1, -1)

    # Model ve etiket yükleme
    model = load_model(model_path)
    label_encoder = joblib.load(label_encoder_path)

    # Tahmin
    prediction = model.predict(features)
    predicted_user = label_encoder.inverse_transform([np.argmax(prediction)])
    print(f"Tahmin Edilen Kullanıcı: {predicted_user[0]}")

if __name__ == "__main__":
    folder_path = 'converted_wav'  # Ses dosyalarının bulunduğu klasör
    csv_path = 'audio_features.csv'  # Özelliklerin kaydedileceği CSV dosyası

    # Model eğitme süreci
    process_data_and_train(folder_path, csv_path)

    # Yeni bir ses dosyasını tahmin etme
    file_path = 'kayit.wav'  # Tahmin yapılacak ses dosyası
    model_path = 'speaker_recognition_model.h5'
    label_encoder_path = 'label_encoder.pkl'
    predict_user(file_path, model_path, label_encoder_path)
