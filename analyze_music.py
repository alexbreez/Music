import librosa
import numpy as np
import sys
import os

def analyze_track(file_path):
    if not os.path.exists(file_path):
        print(f"Ошибка: Файл {file_path} не найден.")
        return

    print(f"--- Анализ файла: {os.path.basename(file_path)} ---")

    # Загрузка аудио (первые 60 сек для скорости)
    y, sr = librosa.load(file_path, duration=60)

    # 1. Определение темпа (BPM)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    print(f"Темп (BPM): {round(float(tempo), 2)}")

    # 2. Определение тональности
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key_map = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key_idx = np.argmax(np.mean(chroma, axis=1))
    print(f"Вероятная тональность: {key_map[key_idx]}")

    # 3. Анализ спектральной яркости (насколько "яркий" звук)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    print(f"Средняя спектральная яркость: {round(float(np.mean(rolloff)), 2)} Гц")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python analyze_music.py <путь_к_файлу>")
    else:
        analyze_track(sys.argv[1])
