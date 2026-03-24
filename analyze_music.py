"""
Анализатор треков v4 — методология «Современная гармония»
=========================================================
Нейросетевое распознавание аккордов (madmom) + функциональный анализ.

Две модели распознавания:
  1. CNN + CRF (madmom CRFChordRecognitionProcessor) — основная
  2. Deep Chroma + CRF (madmom DeepChromaChordRecognitionProcessor) — резервная

Если madmom недоступен, используется DSP-фоллбэк (HPSS + шаблонная корреляция).

Зависимости:
    pip install madmom numpy scipy

Использование:
    python analyze_music.py <путь_к_файлу> [--json] [--method cnn|deepchroma|dsp]
"""

import numpy as np
import subprocess
import json
import sys
import os
import argparse
from collections import Counter

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
ROMAN = {0:'I',1:'bII',2:'II',3:'bIII',4:'III',5:'IV',6:'#IV',7:'V',8:'bVI',9:'VI',10:'bVII',11:'VII'}


# ── НЕЙРОСЕТЕВОЕ РАСПОЗНАВАНИЕ (madmom) ──

def recognize_chords_cnn(file_path):
    from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor
    from madmom.processors import SequentialProcessor
    chordrec = SequentialProcessor([CNNChordFeatureProcessor(), CRFChordRecognitionProcessor()])
    return _parse_madmom(chordrec(file_path))

def recognize_chords_deepchroma(file_path):
    from madmom.audio.chroma import DeepChromaProcessor
    from madmom.features.chords import DeepChromaChordRecognitionProcessor
    from madmom.processors import SequentialProcessor
    chordrec = SequentialProcessor([DeepChromaProcessor(), DeepChromaChordRecognitionProcessor()])
    return _parse_madmom(chordrec(file_path))

def _parse_madmom(result):
    times, chords = [], []
    for entry in result:
        start, end, label = float(entry[0]), float(entry[1]), str(entry[2])
        if label == 'N':
            chord = 'N/C'
        elif ':maj' in label:
            chord = label.split(':')[0]
        elif ':min' in label:
            chord = label.split(':')[0] + 'm'
        else:
            chord = label.replace(':', '')
        times.append(round(start, 2))
        chords.append(chord)
    return times, chords


# ── DSP-ФОЛЛБЭК ──

def recognize_chords_dsp(file_path):
    import scipy.ndimage
    cmd = ['ffmpeg','-i',file_path,'-f','s16le','-acodec','pcm_s16le','-ar','22050','-ac','1','-v','quiet','-']
    y = np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, dtype=np.int16).astype(np.float32)/32768.0
    sr = 22050; n_fft=4096; hop=512
    n_fr = max(1, 1+(len(y)-n_fft)//hop)
    w = np.hanning(n_fft)
    S = np.zeros((n_fft//2+1, n_fr), dtype=np.complex128)
    for i in range(n_fr):
        f = y[i*hop:i*hop+n_fft]
        if len(f)<n_fft: f=np.pad(f,(0,n_fft-len(f)))
        S[:,i] = np.fft.rfft(f*w)
    mag = np.abs(S)
    H = scipy.ndimage.median_filter(mag, size=(1,51))
    P = scipy.ndimage.median_filter(mag, size=(51,1))
    S_h = S*H/(H+P+1e-10)
    freqs = np.fft.rfftfreq(n_fft, 1.0/sr)

    def chroma(t0, t1):
        f0,f1 = int(t0*sr/hop), min(int(t1*sr/hop), S_h.shape[1])
        c = np.zeros(12)
        valid = (freqs>=65)&(freqs<=4200)
        for j in np.where(valid)[0]:
            midi = 12*np.log2(freqs[j]/440.0)+69
            c[int(round(midi))%12] += np.sum(np.abs(S_h[j,f0:f1])**2)
        s = np.linalg.norm(c)
        if s>0: c/=s
        return c

    tmpls = {}
    for r in range(12):
        rn = NOTE_NAMES[r]
        for nm, ivs in [(rn,[r,(r+4)%12,(r+7)%12]), (rn+'m',[r,(r+3)%12,(r+7)%12])]:
            p = np.zeros(12)
            for iv in ivs: p[iv]=1.0
            p /= np.linalg.norm(p)
            tmpls[nm] = p

    # Tempo
    onset = np.zeros(n_fr); prev=None; wn=np.hanning(2048)
    for i in range(n_fr):
        f = y[i*hop:i*hop+2048]
        if len(f)<2048: break
        spec = np.abs(np.fft.rfft(f*wn))
        if prev is not None: onset[i]=np.sum(np.maximum(0, spec[:len(prev)]-prev))
        prev=spec
    corr = np.correlate(onset,onset,mode='full'); corr=corr[len(corr)//2:]
    ml,xl = int(60*sr/hop/200), min(int(60*sr/hop/60), len(corr)-1)
    bl = np.argmax(corr[ml:xl])+ml if xl>ml else int(sr/hop)
    bpm = 60.0*sr/hop/bl; bar = 60.0/bpm*4; dur = len(y)/sr

    times, chords = [], []
    for i in range(int(dur/bar)):
        t0=i*bar; ch=chroma(t0, t0+bar)
        scores = {nm: float(np.dot(ch,t)) for nm,t in tmpls.items()}
        times.append(round(t0,2)); chords.append(max(scores, key=scores.get))
    return times, chords


# ── УТИЛИТЫ ──

def get_root_idx(chord):
    if not chord or chord=='N/C': return None
    if len(chord)>=2 and chord[1] in ('#','b'):
        if chord[:2] in NOTE_NAMES: return NOTE_NAMES.index(chord[:2])
    if chord[0] in 'ABCDEFG': return NOTE_NAMES.index(chord[0])
    return None

def get_quality(chord):
    if not chord or chord=='N/C': return ''
    rl = 2 if (len(chord)>=2 and chord[1] in ('#','b')) else 1
    return chord[rl:]

def simplify(seq):
    if not seq: return []
    r = [seq[0]]
    for c in seq[1:]:
        if c!=r[-1] and c!='N/C': r.append(c)
    return [c for c in r if c!='N/C']


# ── ПАТТЕРН-ПОИСК ──

def find_repeating_patterns(chords, min_len=2, max_len=8, min_reps=2):
    real = [c for c in chords if c!='N/C']
    if len(real)<4: return []
    patterns = []
    for plen in range(min_len, min(max_len+1, len(real)//2+1)):
        for start in range(len(real)-plen*min_reps+1):
            cand = tuple(real[start:start+plen])
            cnt,pos = 0,start
            while pos+plen<=len(real) and tuple(real[pos:pos+plen])==cand:
                cnt+=1; pos+=plen
            if cnt>=min_reps:
                patterns.append({'pattern':list(cand),'repeats':cnt,'start_idx':start,
                    'coverage':round(cnt*plen/len(real)*100,1),'score':cnt*plen})
    seen=set(); unique=[]
    for p in sorted(patterns, key=lambda x:-x['score']):
        k=(tuple(p['pattern']),p['start_idx'])
        if k not in seen: seen.add(k); unique.append(p)
    return unique[:10]


# ── ТОНАЛЬНОСТЬ ──

def detect_key(chords, patterns):
    real = [c for c in chords if c!='N/C']
    if not real: return 'C',0,'major','мажор',0.5
    if patterns:
        last = patterns[0]['pattern'][-1]
        root = get_root_idx(last); q = get_quality(last)
        is_min = q.startswith('m') and 'maj' not in q
        if root is not None:
            m = 'minor' if is_min else 'major'
            return NOTE_NAMES[root],root,m,'минор' if is_min else 'мажор',0.9
    mc = Counter(real).most_common(1)[0][0]
    root = get_root_idx(mc); q = get_quality(mc)
    is_min = q.startswith('m') and 'maj' not in q
    if root is not None:
        m = 'minor' if is_min else 'major'
        return NOTE_NAMES[root],root,m,'минор' if is_min else 'мажор',0.6
    return 'C',0,'major','мажор',0.3


# ── ФУНКЦИОНАЛЬНЫЙ АНАЛИЗ ──

def functional_analysis(progression, key_idx, mode):
    diatonic = MINOR_SCALE if mode=='minor' else MAJOR_SCALE
    result = []
    for chord in progression:
        if chord=='N/C': continue
        root = get_root_idx(chord)
        if root is None: result.append({'chord':chord,'roman':'?','borrowed':False}); continue
        iv = (root-key_idx)%12
        rb = ROMAN.get(iv,'?'); q = get_quality(chord)
        if 'm7b5' in q: d=rb.lower()+'m7b5'
        elif 'm7' in q: d=rb.lower()+'7'
        elif q=='m': d=rb.lower()
        elif 'maj7' in q: d=rb+'maj7'
        elif '7' in q: d=rb+'7'
        elif 'dim' in q: d=rb.lower()+'°'
        elif 'aug' in q: d=rb+'+'
        else: d=rb
        result.append({'chord':chord,'roman':d,'borrowed':iv not in diatonic})
    return result


# ── ПАТТЕРНЫ «СОВРЕМЕННОЙ ГАРМОНИИ» ──

def detect_harmony_patterns(analysis):
    patterns = []; chords=[a['chord'] for a in analysis]; romans=[a['roman'] for a in analysis]
    # Пикардийская терция
    roots={}
    for c in chords:
        r=get_root_idx(c); q=get_quality(c)
        if r is None: continue
        if r not in roots: roots[r]={'min':None,'maj':None}
        if q.startswith('m') and 'maj' not in q: roots[r]['min']=c
        if q in ('','maj7','7'): roots[r]['maj']=c
    for info in roots.values():
        if info['min'] and info['maj']:
            patterns.append({'type':'Пикардийская терция','description':f"{info['min']} → {info['maj']}"})
    # Рок-каденция
    for i in range(len(romans)-2):
        if 'bVI' in romans[i] and 'bVII' in romans[i+1] and romans[i+2] in ('I','i'):
            patterns.append({'type':'Рок-каденция','description':f'{chords[i]} → {chords[i+1]} → {chords[i+2]}'})
    # Обманная каденция
    for i in range(len(romans)-1):
        if romans[i] in ('V','V7') and 'bVI' in romans[i+1]:
            patterns.append({'type':'Обманная каденция','description':f'{chords[i]} → {chords[i+1]}'})
    # Полууменьшенные
    for c in set(chords):
        if 'm7b5' in c: patterns.append({'type':'Полууменьшенный аккорд','description':f'{c}'})
    # Заимствованные
    bor = sorted(set(f"{a['chord']} ({a['roman']})" for a in analysis if a['borrowed']))
    if bor: patterns.append({'type':'Заимствованные аккорды','description':',  '.join(bor)})
    return patterns


# ── ДВИЖЕНИЯ ──

def find_movements(chords, chord_times, rep_patterns):
    real = [(t,c) for t,c in zip(chord_times, chords) if c!='N/C']
    if not real or not rep_patterns: return []
    co = [r[1] for r in real]; movements=[]; used=set()
    for pi in rep_patterns[:3]:
        pat=pi['pattern']; pl=len(pat); i=0
        while i<=len(co)-pl:
            if i in used: i+=1; continue
            if co[i:i+pl]==pat:
                si=i; reps=0
                while i+pl<=len(co) and co[i:i+pl]==pat:
                    for j in range(i,i+pl): used.add(j)
                    reps+=1; i+=pl
                movements.append({'pattern':pat,'repeats':reps,'start_time':real[si][0],'end_time':real[min(i-1,len(real)-1)][0]})
            else: i+=1
    return sorted(movements, key=lambda m:m['start_time'])


# ── ТЕМП ──

def detect_tempo(file_path, sr=22050):
    cmd = ['ffmpeg','-i',file_path,'-f','s16le','-acodec','pcm_s16le','-ar',str(sr),'-ac','1','-v','quiet','-']
    y = np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, dtype=np.int16).astype(np.float32)/32768.0
    dur = len(y)/sr; hop=512; n_fft=2048; n_fr=max(1,1+(len(y)-n_fft)//hop)
    onset=np.zeros(n_fr); prev=None; w=np.hanning(n_fft)
    for i in range(n_fr):
        f=y[i*hop:i*hop+n_fft]
        if len(f)<n_fft: break
        s=np.abs(np.fft.rfft(f*w))
        if prev is not None: onset[i]=np.sum(np.maximum(0,s[:len(prev)]-prev))
        prev=s
    corr=np.correlate(onset,onset,mode='full'); corr=corr[len(corr)//2:]
    ml,xl=int(60*sr/hop/200),min(int(60*sr/hop/60),len(corr)-1)
    bl=np.argmax(corr[ml:xl])+ml if xl>ml else int(sr/hop)
    return round(60.0*sr/hop/bl,1), round(dur,2)


# ── MAIN ──

def analyze_track(file_path, method='auto'):
    if not os.path.exists(file_path): print(f"Ошибка: {file_path} не найден."); return None
    fn = os.path.basename(file_path)
    print(f"\n{'='*64}\n  АНАЛИЗ v4: {fn}\n  Методология «Современная гармония»\n{'='*64}")

    bpm, dur = detect_tempo(file_path)
    print(f"\n🥁 Темп: {bpm} BPM, {int(dur//60)}:{int(dur%60):02d}")

    ct, ch, um = None, None, method
    if method in ('auto','cnn'):
        try: print("\n🧠 CNN + CRF..."); ct,ch=recognize_chords_cnn(file_path); um='cnn'; print("   ✓ madmom CNN")
        except ImportError:
            if method=='cnn': print("   ✗ madmom не установлен"); return None
    if ct is None and method in ('auto','deepchroma'):
        try: print("\n🧠 Deep Chroma + CRF..."); ct,ch=recognize_chords_deepchroma(file_path); um='deepchroma'; print("   ✓ madmom DeepChroma")
        except ImportError:
            if method=='deepchroma': print("   ✗ madmom не установлен"); return None
    if ct is None:
        print("\n🔬 DSP-фоллбэк..."); ct,ch=recognize_chords_dsp(file_path); um='dsp'
        print("   ⚠ DSP (рекомендуется pip install madmom)")

    real = [c for c in ch if c!='N/C']
    prog = simplify(ch)
    print(f"\n🎸 Уникальных: {len(set(real))}, метод: {um}")
    print(f"   {' → '.join(prog[:30])}")

    rp = find_repeating_patterns(real)
    if rp:
        print(f"\n🔁 Паттерны:")
        for p in rp[:5]: print(f"   {' → '.join(p['pattern'])} (×{p['repeats']}, {p['coverage']}%)")

    kn,ki,mo,mr,kc = detect_key(real, rp)
    print(f"\n🎵 Тональность: {kn} {mr} ({kc})")

    an = functional_analysis(prog, ki, mo)
    print(f"\n📊 Ступени: {' → '.join([a['roman'] for a in an[:30]])}")
    bor = [a for a in an if a['borrowed']]
    if bor: print(f"   ⚡ Заимств.: {', '.join(sorted(set(f'{a[\"chord\"]} ({a[\"roman\"]})' for a in bor)))}")

    mv = find_movements(ch, ct, rp)
    if mv:
        print(f"\n📐 Движения:")
        for i,m in enumerate(mv,1): print(f"   {i}: {' → '.join(m['pattern'])} (×{m['repeats']}, {m['start_time']:.1f}-{m['end_time']:.1f}s)")

    hp = detect_harmony_patterns(an)
    if hp:
        print(f"\n🔍 Паттерны «Совр. гармонии»:")
        for p in hp: print(f"   ✦ {p['type']}: {p['description']}")

    mp = rp[0]['pattern'] if rp else prog[:4]
    print(f"\n{'='*64}\n  ИТОГ: {kn} {mr} | {bpm} BPM | {' → '.join(mp)} | {um}\n{'='*64}\n")
    return {'filename':fn,'duration':dur,'tempo_bpm':bpm,'method':um,'key':kn,'mode':mr,
        'main_pattern':mp,'progression':prog,'chords':ch,'chord_times':ct,
        'functional':[{'chord':a['chord'],'roman':a['roman'],'borrowed':a['borrowed']} for a in an],
        'movements':mv,'harmony_patterns':hp}

def main():
    p = argparse.ArgumentParser(description='Анализатор v4')
    p.add_argument('file'); p.add_argument('--method',choices=['auto','cnn','deepchroma','dsp'],default='auto')
    p.add_argument('--json',action='store_true')
    a = p.parse_args()
    r = analyze_track(a.file, method=a.method)
    if a.json and r: print(json.dumps(r, ensure_ascii=False, indent=2, default=str))

if __name__=='__main__': main()
