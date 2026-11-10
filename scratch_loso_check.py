
import os, re, glob, mne, numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

EEG_CHANNELS = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']

def moving_average(signal, w=5):
    k = np.ones(w)/w
    out = np.zeros_like(signal)
    for i in range(signal.shape[0]):
        out[i] = np.convolve(signal[i], k, mode='same')
    return out

def paper_feat(seg):
    ddof = 1 if seg.shape[1] > 1 else 0
    return np.concatenate([
        np.std(seg, axis=1, ddof=ddof),
        np.sqrt(np.mean(seg**2, axis=1)),
        np.sum(seg, axis=1),
        np.sum(seg**2, axis=1)
    ])

# Load all records
files = sorted(glob.glob('./data/Digit/*.edf'))
records = []
print("Loading dataset...")
for f in files:
    name = os.path.basename(f)
    m = re.fullmatch(r'name(\d+)_(\d+)\.edf', name, re.IGNORECASE)
    p, l = int(m.group(1)), int(m.group(2))
    raw = mne.io.read_raw_edf(f, preload=True, verbose='ERROR')
    raw.pick(EEG_CHANNELS)
    data = moving_average(raw.get_data())
    sr = raw.info['sfreq']
    win = int(0.05 * sr)
    feats = []
    idx = 0
    while idx + win <= data.shape[1]:
        feats.append(paper_feat(data[:, idx:idx+win]))
        idx += win
    records.append({'p': p, 'l': l, 'f': np.array(feats), 'file': name})

participants = sorted(set(r['p'] for r in records))
print(f"Loaded {len(records)} recordings, {len(participants)} participants\n")

# Quick LOSO
print("Participant   Accuracy   Correctly-predicted digits")
print("-" * 55)

results = {}
for p in participants:
    train = [r for r in records if r['p'] != p]
    test  = [r for r in records if r['p'] == p]

    X_tr = np.vstack([r['f'] for r in train])
    y_tr = np.concatenate([np.full(len(r['f']), r['l']) for r in train])

    rf = RandomForestClassifier(n_estimators=40, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    preds = {}
    correct = 0
    for rec in test:
        wp = rf.predict(rec['f'])
        c  = Counter(wp.tolist())
        v  = min(k for k, val in c.items() if val == max(c.values()))
        preds[rec['l']] = v
        if v == rec['l']:
            correct += 1

    acc = correct / len(test)
    good = sorted([d for d, pred in preds.items() if pred == d])
    results[p] = {'acc': acc, 'preds': preds, 'good': good}
    print(f"  P{p:<10}  {acc:>7.0%}    {good}")

best_p = max(results, key=lambda p: results[p]['acc'])
print(f"\nBest leave-out candidate : Participant {best_p}")
print(f"Accuracy on their files  : {results[best_p]['acc']:.0%}")
print(f"Correctly predicted      : {results[best_p]['good']}")
print(f"All digit predictions    : {results[best_p]['preds']}")
