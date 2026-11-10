
"""
Find the best individual files to use in the class demo.
Tests every file in the dataset using the saved model and
reports which are predicted correctly with highest confidence.
"""

import os, re, glob, mne, numpy as np, joblib
from collections import Counter

EEG_CHANNELS = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']

def moving_average(signal, w=5):
    k = np.ones(w) / w
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

# Load model
artifact = joblib.load('./eeg_digit_rf.joblib')
model = artifact['model']

files = sorted(glob.glob('./data/Digit/*.edf'))

print("Testing every file against the saved model...")
print(f"\n{'File':<20} {'True':>5} {'Pred':>5} {'Correct':>8} {'Confidence':>12} {'Vote%':>7}")
print("-" * 65)

results = []

for f in files:
    name = os.path.basename(f)
    m    = re.fullmatch(r'name(\d+)_(\d+)\.edf', name, re.IGNORECASE)
    p, l = int(m.group(1)), int(m.group(2))

    raw = mne.io.read_raw_edf(f, preload=True, verbose='ERROR')
    raw.pick(EEG_CHANNELS)
    data = moving_average(raw.get_data())
    sr   = raw.info['sfreq']
    win  = int(0.05 * sr)

    feats = []
    idx = 0
    while idx + win <= data.shape[1]:
        feats.append(paper_feat(data[:, idx:idx+win]))
        idx += win

    X = np.array(feats)
    preds = model.predict(X)
    probs = model.predict_proba(X)

    c    = Counter(preds.tolist())
    vote = min(k for k, v in c.items() if v == max(c.values()))
    conf = probs.mean(axis=0)[vote] * 100
    vote_pct = c[vote] / len(preds) * 100
    correct  = vote == l

    results.append({
        'file': name, 'participant': p, 'true': l,
        'pred': vote, 'correct': correct,
        'confidence': conf, 'vote_pct': vote_pct
    })

    tick = "YES" if correct else "NO"
    print(f"{name:<20} {l:>5} {vote:>5} {tick:>8} {conf:>11.1f}% {vote_pct:>6.1f}%")

# Summary
correct_all = [r for r in results if r['correct']]
print(f"\n{'='*65}")
print(f"Total files      : {len(results)}")
print(f"Correctly predicted : {len(correct_all)} / {len(results)}  ({len(correct_all)/len(results):.0%})")

print(f"\n{'TOP 20 FILES FOR DEMO (correct + highest confidence)':}")
print(f"{'─'*65}")
top = sorted(correct_all, key=lambda r: r['confidence'], reverse=True)[:20]
print(f"{'File':<20} {'Digit':>6} {'Confidence':>12} {'Vote%':>8}")
print(f"{'─'*48}")
for r in top:
    print(f"{r['file']:<20} {r['true']:>6} {r['confidence']:>11.1f}% {r['vote_pct']:>7.1f}%")

print(f"\n{'BEST 3 FILES (use these for demo)':}")
print(f"{'─'*65}")
for r in top[:3]:
    print(f"  {r['file']}  →  Digit {r['true']}  |  Confidence: {r['confidence']:.1f}%  |  Votes: {r['vote_pct']:.1f}%")
