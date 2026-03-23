"""
Анализатор треков по методологии «Современная гармония»
=====================================================
Расширенная версия: распознавание аккордов, секций, мелодических акцентов.

Зависимости:
    pip install librosa numpy soundfile madmom vamp

Использование:
    python analyze_music.py <путь_к_аудиофайлу> [--full] [--json]

Флаги:
    --full   Полный анализ (включая мелодические акценты, медленнее)
    --json   Вывод в формате JSON (для интеграции с UI)
"""

import librosa
import librosa.display
import numpy as np
import sys
import os
import json
import argparse
from collections import Counter

# ─────────────────────────────────────────────
# 1. КОНСТАНТЫ И МУЗЫКАЛЬНАЯ ТЕОРИЯ
# ─────────────────────────────────────────────

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Шаблоны аккордов (интервалы от корня в полутонах)
CHORD_TEMPLATES = {
    'maj':      [0, 4, 7],
    'min':      [0, 3, 7],
    '7':        [0, 4, 7, 10],
    'maj7':     [0, 4, 7, 11],
    'min7':     [0, 3, 7, 10],
    'dim':      [0, 3, 6],
    'aug':      [0, 4, 8],
    'sus2':     [0, 2, 7],
    'sus4':     [0, 5, 7],
    'min7b5':   [0, 3, 6, 10],   # полууменьшенный
    'dim7':     [0, 3, 6, 9],
    'add9':     [0, 4, 7, 14],
    '9':        [0, 4, 7, 10, 14],
}

# Ступени мажорного лада (интервалы в полутонах)
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
# Ступени натурального минорного лада
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
# Миксолидийский лад
MIXOLYDIAN_SCALE = [0, 2, 4, 5, 7, 9, 10]
# Дорийский лад
DORIAN_SCALE = [0, 2, 3, 5, 7, 9, 10]
# Лидийский лад
LYDIAN_SCALE = [0, 2, 4, 6, 7, 9, 11]

SCALE_PROFILES = {
    'мажор (ионийский)': MAJOR_SCALE,
    'натуральный минор (эолийский)': MINOR_SCALE,
    'миксолидийский': MIXOLYDIAN_SCALE,
    'дорийский': DORIAN_SCALE,
    'лидийский': LYDIAN_SCALE,
}

# Римские цифры для ступеней
ROMAN_MAJOR = {
    0: 'I', 1: 'bII', 2: 'II', 3: 'bIII', 4: 'III', 5: 'IV',
    6: '#IV/bV', 7: 'V', 8: 'bVI', 9: 'VI', 10: 'bVII', 11: 'VII'
}
ROMAN_MINOR = {
    0: 'i', 1: 'bII', 2: 'II', 3: 'bIII/III', 4: 'IV', 5: 'iv',
    6: '#iv/bV', 7: 'v/V', 8: 'bVI/VI', 9: 'bVII', 10: 'VII', 11: 'vii'
}


# ─────────────────────────────────────────────
# 2. ОПРЕДЕЛЕНИЕ ТЕМПА И ТОНАЛЬНОСТИ
# ─────────────────────────────────────────────

def detect_tempo(y, sr):
    """Определение темпа (BPM)."""
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return float(tempo), beat_times


def detect_key_and_mode(y, sr):
    """
    Определение тональности и лада через корреляцию хромаграммы
    с профилями мажора/минора по Крумхансл-Шмуклер.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_mean = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-6)

    # Профили Крумхансл-Шмуклер
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    major_profile = major_profile / np.linalg.norm(major_profile)
    minor_profile = minor_profile / np.linalg.norm(minor_profile)

    best_corr = -1
    best_key = 0
    best_mode = 'major'

    for shift in range(12):
        shifted = np.roll(chroma_mean, -shift)
        corr_maj = np.corrcoef(shifted, major_profile)[0, 1]
        corr_min = np.corrcoef(shifted, minor_profile)[0, 1]

        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key = shift
            best_mode = 'major'
        if corr_min > best_corr:
            best_corr = corr_min
            best_key = shift
            best_mode = 'minor'

    key_name = NOTE_NAMES[best_key]
    mode_ru = 'мажор' if best_mode == 'major' else 'минор'
    confidence = round(best_corr, 3)

    return key_name, best_mode, mode_ru, confidence


def detect_scale_type(chords, key_idx):
    """
    Определяет вероятный лад (ионийский, миксолидийский, дорийский, лидийский)
    на основе реально встречающихся аккордов.
    """
    chord_roots = []
    for chord_name in chords:
        root = chord_name.split(' ')[0] if ' ' in chord_name else chord_name.replace('m', '').replace('7', '').replace('b5', '').replace('dim', '').replace('aug', '').replace('sus', '').replace('add', '')
        # Упрощённое извлечение корня
        for i, name in enumerate(NOTE_NAMES):
            if chord_name.startswith(name) and (len(chord_name) == len(name) or not chord_name[len(name)].isalpha() or chord_name[len(name)] == 'm'):
                chord_roots.append((i - key_idx) % 12)
                break

    root_set = set(chord_roots)
    best_match = 'мажор (ионийский)'
    best_score = 0

    for scale_name, intervals in SCALE_PROFILES.items():
        score = len(root_set.intersection(set(intervals)))
        if score > best_score:
            best_score = score
            best_match = scale_name

    return best_match


# ─────────────────────────────────────────────
# 3. РАСПОЗНАВАНИЕ АККОРДОВ
# ─────────────────────────────────────────────

def build_chord_templates():
    """Строит матрицу шаблонов аккордов (12 корней × N типов) → хрома-профили."""
    templates = []
    labels = []
    for root in range(12):
        for chord_type, intervals in CHORD_TEMPLATES.items():
            profile = np.zeros(12)
            for interval in intervals:
                profile[(root + interval) % 12] = 1.0
            profile = profile / (np.linalg.norm(profile) + 1e-6)
            templates.append(profile)

            suffix = '' if chord_type == 'maj' else chord_type
            if chord_type == 'min':
                suffix = 'm'
            elif chord_type == 'min7':
                suffix = 'm7'
            elif chord_type == 'min7b5':
                suffix = 'm7b5'

            labels.append(f"{NOTE_NAMES[root]}{suffix}")

    return np.array(templates), labels


def recognize_chords(y, sr, beat_times=None, hop_length=512):
    """
    Распознавание аккордов по хромаграмме.
    Использует корреляцию с шаблонами аккордов.
    Возвращает список (время, аккорд) и упрощённую прогрессию.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    templates, labels = build_chord_templates()

    if beat_times is not None and len(beat_times) > 1:
        beat_frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)
        beat_frames = beat_frames[beat_frames < chroma.shape[1]]
    else:
        # Если нет битов — разбиваем на фреймы по ~0.5 сек
        n_frames = chroma.shape[1]
        step = max(1, int(0.5 * sr / hop_length))
        beat_frames = np.arange(0, n_frames, step)

    chord_sequence = []
    chord_times = []

    for i in range(len(beat_frames) - 1):
        start_f = beat_frames[i]
        end_f = beat_frames[i + 1]
        segment = chroma[:, start_f:end_f]
        if segment.shape[1] == 0:
            continue

        avg_chroma = np.mean(segment, axis=1)
        avg_chroma = avg_chroma / (np.linalg.norm(avg_chroma) + 1e-6)

        correlations = templates @ avg_chroma
        best_idx = np.argmax(correlations)
        confidence = correlations[best_idx]

        if confidence > 0.5:
            chord_sequence.append(labels[best_idx])
            t = librosa.frames_to_time(start_f, sr=sr, hop_length=hop_length)
            chord_times.append(round(float(t), 2))

    return chord_times, chord_sequence


def simplify_progression(chord_sequence):
    """
    Убирает подряд идущие дубликаты, возвращает чистую прогрессию.
    """
    if not chord_sequence:
        return []
    simplified = [chord_sequence[0]]
    for chord in chord_sequence[1:]:
        if chord != simplified[-1]:
            simplified.append(chord)
    return simplified


# ─────────────────────────────────────────────
# 4. ФУНКЦИОНАЛЬНЫЙ АНАЛИЗ (СТУПЕНИ)
# ─────────────────────────────────────────────

def chord_to_roman(chord_name, key_idx, mode='minor'):
    """
    Преобразует название аккорда в римскую цифру (ступень) относительно тональности.
    """
    # Извлекаем корень
    root_idx = None
    for i, name in enumerate(NOTE_NAMES):
        if chord_name.startswith(name):
            if len(chord_name) == len(name) or not chord_name[len(name)].isalpha() or chord_name[len(name)] in 'ms7db':
                root_idx = i
                break

    if root_idx is None:
        return chord_name

    interval = (root_idx - key_idx) % 12
    roman = ROMAN_MINOR.get(interval, '?') if mode == 'minor' else ROMAN_MAJOR.get(interval, '?')

    # Определяем качество аккорда для уточнения
    quality = chord_name[len(NOTE_NAMES[root_idx]):]
    if 'm7b5' in quality:
        roman = roman.lower() + '°7'
    elif 'dim' in quality:
        roman = roman.lower() + '°'
    elif 'aug' in quality:
        roman = roman.upper() + '+'
    elif 'm7' in quality:
        roman = roman.lower() + '7'
    elif 'm' in quality and 'maj' not in quality:
        roman = roman.lower()
    elif '7' in quality:
        roman = roman.upper() + '7'
    elif 'sus' in quality:
        roman = roman + 'sus'
    elif 'maj7' in quality:
        roman = roman.upper() + 'maj7'

    return roman


def functional_analysis(progression, key_idx, mode='minor'):
    """
    Возвращает прогрессию в римских цифрах + выделяет заимствованные аккорды.
    """
    if mode == 'minor':
        diatonic_intervals = MINOR_SCALE
    else:
        diatonic_intervals = MAJOR_SCALE

    analysis = []
    for chord in progression:
        roman = chord_to_roman(chord, key_idx, mode)

        # Определяем, является ли аккорд диатоническим
        root_idx = None
        for i, name in enumerate(NOTE_NAMES):
            if chord.startswith(name):
                if len(chord) == len(name) or not chord[len(name)].isalpha() or chord[len(name)] in 'ms7db':
                    root_idx = i
                    break

        is_borrowed = False
        if root_idx is not None:
            interval = (root_idx - key_idx) % 12
            if interval not in diatonic_intervals:
                is_borrowed = True

        analysis.append({
            'chord': chord,
            'roman': roman,
            'borrowed': is_borrowed,
        })

    return analysis


# ─────────────────────────────────────────────
# 5. СЕГМЕНТАЦИЯ НА СЕКЦИИ
# ─────────────────────────────────────────────

def detect_sections(y, sr):
    """
    Сегментация трека на секции (куплет, припев, бридж и т.д.)
    через Self-Similarity Matrix + спектральную кластеризацию.
    """
    # Извлекаем MFCC для представления тембра
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # Извлекаем хрому для гармонии
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    # Объединяем признаки
    features = np.vstack([
        librosa.util.normalize(mfcc, axis=1),
        librosa.util.normalize(chroma, axis=1)
    ])

    # Вычисляем границы секций через novelty function
    try:
        bounds = librosa.segment.agglomerative(features, k=None)
        bound_times = librosa.frames_to_time(bounds, sr=sr)
    except Exception:
        # Фоллбэк: используем спектральный поток
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        bounds = librosa.segment.agglomerative(
            librosa.feature.mfcc(y=y, sr=sr), k=6
        )
        bound_times = librosa.frames_to_time(bounds, sr=sr)

    # Кластеризуем секции по схожести
    n_sections = len(bound_times) - 1 if len(bound_times) > 1 else 1
    sections = []

    for i in range(min(n_sections, len(bound_times))):
        start = bound_times[i] if i < len(bound_times) else 0
        end = bound_times[i + 1] if (i + 1) < len(bound_times) else librosa.get_duration(y=y, sr=sr)

        # Вычисляем среднюю энергию секции для классификации
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = y[start_sample:end_sample]

        if len(segment) > 0:
            rms = float(np.sqrt(np.mean(segment ** 2)))
        else:
            rms = 0.0

        sections.append({
            'start': round(float(start), 2),
            'end': round(float(end), 2),
            'duration': round(float(end - start), 2),
            'energy': round(rms, 4),
            'label': f'Секция {i + 1}',  # предварительная метка
        })

    # Классифицируем секции по энергии (грубая эвристика)
    if sections:
        energies = [s['energy'] for s in sections]
        median_energy = np.median(energies)

        for s in sections:
            if s['energy'] > median_energy * 1.2:
                s['label_guess'] = 'припев (высокая энергия)'
            elif s['energy'] < median_energy * 0.8:
                s['label_guess'] = 'куплет/бридж (низкая энергия)'
            else:
                s['label_guess'] = 'переходная секция'

    return sections


# ─────────────────────────────────────────────
# 6. МЕЛОДИЧЕСКИЕ АКЦЕНТЫ
# ─────────────────────────────────────────────

def detect_melody_accents(y, sr, chord_times, chord_sequence, key_idx):
    """
    Определяет мелодические акценты: какие ноты мелодии звучат на каких аккордах,
    вычисляет ступень мелодии относительно аккорда и тональности.
    """
    # Извлекаем основную высоту тона (мелодию) через piptrack
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    accents = []

    for i in range(len(chord_times) - 1):
        t_start = chord_times[i]
        t_end = chord_times[i + 1] if (i + 1) < len(chord_times) else t_start + 1.0

        # Конвертируем время в фреймы
        f_start = librosa.time_to_frames(t_start, sr=sr)
        f_end = librosa.time_to_frames(t_end, sr=sr)
        f_end = min(f_end, pitches.shape[1])

        if f_start >= f_end:
            continue

        # Находим доминирующую высоту в этом отрезке
        segment_pitches = pitches[:, f_start:f_end]
        segment_mags = magnitudes[:, f_start:f_end]

        # Берём самые громкие ноты
        top_pitches = []
        for frame in range(segment_pitches.shape[1]):
            idx = segment_mags[:, frame].argmax()
            pitch = segment_pitches[idx, frame]
            if pitch > 0:
                top_pitches.append(pitch)

        if not top_pitches:
            continue

        # Медиана — приблизительная нота мелодии
        median_pitch = np.median(top_pitches)
        if median_pitch <= 0:
            continue

        # Переводим частоту в MIDI → название ноты
        midi_note = librosa.hz_to_midi(median_pitch)
        note_idx = int(round(midi_note)) % 12
        note_name = NOTE_NAMES[note_idx]

        # Ступень относительно корня аккорда
        chord_name = chord_sequence[i] if i < len(chord_sequence) else ''
        chord_root_idx = None
        for j, name in enumerate(NOTE_NAMES):
            if chord_name.startswith(name):
                if len(chord_name) == len(name) or not chord_name[len(name)].isalpha() or chord_name[len(name)] in 'ms7db':
                    chord_root_idx = j
                    break

        if chord_root_idx is not None:
            interval_from_chord = (note_idx - chord_root_idx) % 12
            interval_from_key = (note_idx - key_idx) % 12

            # Названия интервалов
            INTERVAL_NAMES = {
                0: '1 (прима)', 1: 'b9', 2: '9', 3: 'b3/m3', 4: '3',
                5: '11', 6: '#11/b5', 7: '5', 8: 'b13/#5', 9: '13/b7(maj)',
                10: 'b7', 11: '7+'
            }

            accents.append({
                'time': t_start,
                'chord': chord_name,
                'melody_note': note_name,
                'interval_from_chord': INTERVAL_NAMES.get(interval_from_chord, '?'),
                'interval_from_key': INTERVAL_NAMES.get(interval_from_key, '?'),
            })

    return accents


# ─────────────────────────────────────────────
# 7. ОБНАРУЖЕНИЕ ПАТТЕРНОВ «СОВРЕМЕННОЙ ГАРМОНИИ»
# ─────────────────────────────────────────────

def detect_harmony_patterns(analysis):
    """
    Ищет характерные паттерны из методологии «Современная гармония»:
    - Рок-каденция (bVI - bVII - I)
    - Обманная каденция (V → bVI вместо I)
    - Полууменьшенные аккорды (m7b5) с задержанным разрешением
    - Лидийский аккорд
    - Заимствованные аккорды
    """
    patterns = []

    romans = [a['roman'] for a in analysis]
    chords = [a['chord'] for a in analysis]

    # Рок-каденция: bVI - bVII - I
    for i in range(len(romans) - 2):
        if 'bVI' in romans[i] and 'bVII' in romans[i + 1] and romans[i + 2] in ('I', 'i'):
            patterns.append({
                'type': 'Рок-каденция',
                'description': f'bVI → bVII → I ({chords[i]} → {chords[i+1]} → {chords[i+2]})',
                'position': i,
            })

    # Обманная каденция: V → bVI
    for i in range(len(romans) - 1):
        if romans[i] in ('V', 'V7') and 'bVI' in romans[i + 1]:
            patterns.append({
                'type': 'Обманная каденция',
                'description': f'V → bVI ({chords[i]} → {chords[i+1]})',
                'position': i,
            })

    # Полууменьшенные аккорды
    for i, chord in enumerate(chords):
        if 'm7b5' in chord:
            patterns.append({
                'type': 'Полууменьшенный аккорд',
                'description': f'{chord} (позиция {i+1}) — задержанное разрешение',
                'position': i,
            })

    # Заимствованные аккорды
    borrowed = [a for a in analysis if a['borrowed']]
    if borrowed:
        borrowed_names = [f"{a['chord']} ({a['roman']})" for a in borrowed]
        patterns.append({
            'type': 'Заимствованные аккорды',
            'description': ', '.join(borrowed_names),
            'position': -1,
        })

    return patterns


# ─────────────────────────────────────────────
# 8. СПЕКТРАЛЬНЫЙ АНАЛИЗ (из оригинала + расширения)
# ─────────────────────────────────────────────

def spectral_analysis(y, sr):
    """Спектральная яркость и динамический контраст."""
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]

    return {
        'spectral_rolloff_mean': round(float(np.mean(rolloff)), 2),
        'spectral_rolloff_std': round(float(np.std(rolloff)), 2),
        'dynamic_range_db': round(float(20 * np.log10(np.max(rms) / (np.min(rms) + 1e-6))), 2),
        'rms_mean': round(float(np.mean(rms)), 4),
    }


# ─────────────────────────────────────────────
# 9. ГЛАВНАЯ ФУНКЦИЯ АНАЛИЗА
# ─────────────────────────────────────────────

def analyze_track(file_path, full_analysis=False):
    """
    Полный анализ трека.
    Возвращает словарь с результатами.
    """
    if not os.path.exists(file_path):
        return {'error': f'Файл {file_path} не найден.'}

    filename = os.path.basename(file_path)
    print(f"\n{'='*60}")
    print(f"  АНАЛИЗ: {filename}")
    print(f"  Методология «Современная гармония»")
    print(f"{'='*60}\n")

    # Загрузка аудио
    print("⏳ Загрузка аудио...")
    y, sr = librosa.load(file_path, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"   Длительность: {int(duration // 60)}:{int(duration % 60):02d}")

    result = {
        'filename': filename,
        'duration': round(duration, 2),
    }

    # --- Темп ---
    print("\n🎵 Определение темпа...")
    tempo, beat_times = detect_tempo(y, sr)
    result['tempo_bpm'] = round(tempo, 1)
    print(f"   Темп: {result['tempo_bpm']} BPM")

    # --- Тональность ---
    print("\n🎹 Определение тональности...")
    key_name, mode, mode_ru, key_confidence = detect_key_and_mode(y, sr)
    key_idx = NOTE_NAMES.index(key_name)
    result['key'] = key_name
    result['mode'] = mode_ru
    result['key_confidence'] = key_confidence
    print(f"   Тональность: {key_name} {mode_ru} (уверенность: {key_confidence})")

    # --- Распознавание аккордов ---
    print("\n🎸 Распознавание аккордов...")
    chord_times, chord_sequence = recognize_chords(y, sr, beat_times)
    progression = simplify_progression(chord_sequence)
    result['raw_chord_count'] = len(chord_sequence)
    result['progression'] = progression
    print(f"   Уникальных аккордов: {len(set(chord_sequence))}")
    print(f"   Прогрессия: {' → '.join(progression[:16])}{'...' if len(progression) > 16 else ''}")

    # --- Определение лада ---
    scale_type = detect_scale_type(chord_sequence, key_idx)
    result['scale_type'] = scale_type
    print(f"   Вероятный лад: {scale_type}")

    # --- Функциональный анализ ---
    print("\n📊 Функциональный анализ ступеней...")
    analysis = functional_analysis(progression, key_idx, mode)
    result['functional_analysis'] = analysis

    roman_str = ' → '.join([a['roman'] for a in analysis[:16]])
    print(f"   Ступени: {roman_str}{'...' if len(analysis) > 16 else ''}")

    borrowed = [a for a in analysis if a['borrowed']]
    if borrowed:
        print(f"   ⚡ Заимствованные: {', '.join([f'{a[\"chord\"]} ({a[\"roman\"]})' for a in borrowed])}")

    # --- Паттерны «Современной гармонии» ---
    print("\n🔍 Поиск паттернов «Современной гармонии»...")
    patterns = detect_harmony_patterns(analysis)
    result['patterns'] = patterns

    if patterns:
        for p in patterns:
            print(f"   ✦ {p['type']}: {p['description']}")
    else:
        print("   Характерных паттернов не обнаружено")

    # --- Секции ---
    print("\n📐 Сегментация на секции...")
    sections = detect_sections(y, sr)
    result['sections'] = sections
    for s in sections:
        guess = s.get('label_guess', '')
        print(f"   [{s['start']:.1f}s - {s['end']:.1f}s] {s['label']} — {guess} (энергия: {s['energy']})")

    # --- Спектральный анализ ---
    print("\n🌈 Спектральный анализ...")
    spectral = spectral_analysis(y, sr)
    result['spectral'] = spectral
    print(f"   Средняя яркость: {spectral['spectral_rolloff_mean']} Гц")
    print(f"   Динамический диапазон: {spectral['dynamic_range_db']} дБ")

    # --- Мелодические акценты (опционально) ---
    if full_analysis:
        print("\n🎤 Анализ мелодических акцентов (это может занять время)...")
        accents = detect_melody_accents(y, sr, chord_times, chord_sequence, key_idx)
        result['melody_accents'] = accents

        if accents:
            print(f"   Найдено {len(accents)} мелодических акцентов")
            for a in accents[:10]:
                print(f"   [{a['time']:.1f}s] {a['melody_note']} на {a['chord']} "
                      f"→ от аккорда: {a['interval_from_chord']}, от тоники: {a['interval_from_key']}")
            if len(accents) > 10:
                print(f"   ... и ещё {len(accents) - 10}")

    # --- Итог ---
    print(f"\n{'='*60}")
    print(f"  ИТОГ АНАЛИЗА")
    print(f"{'='*60}")
    print(f"  Трек: {filename}")
    print(f"  Тональность: {key_name} {mode_ru} ({scale_type})")
    print(f"  Темп: {result['tempo_bpm']} BPM")
    print(f"  Секций: {len(sections)}")
    print(f"  Паттернов: {len(patterns)}")
    if borrowed:
        print(f"  Заимствованных аккордов: {len(borrowed)}")
    print(f"{'='*60}\n")

    return result


# ─────────────────────────────────────────────
# 10. CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Анализатор треков по методологии «Современная гармония»'
    )
    parser.add_argument('file', help='Путь к аудиофайлу (mp3, wav, flac)')
    parser.add_argument('--full', action='store_true',
                        help='Полный анализ (включая мелодические акценты)')
    parser.add_argument('--json', action='store_true',
                        help='Вывод результатов в формате JSON')

    args = parser.parse_args()
    result = analyze_track(args.file, full_analysis=args.full)

    if args.json:
        # Убираем непечатаемые данные для JSON
        clean = {k: v for k, v in result.items()}
        print(json.dumps(clean, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
