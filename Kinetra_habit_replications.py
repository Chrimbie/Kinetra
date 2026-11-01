#!/usr/bin/env python3
"""Independent training/starting-track replications, deterministic, no test-set tuning."""
from pathlib import Path
import csv
import numpy as np
from Kinetra_habit_corrections import tracks, cases, train_starts, test_starts, learn, rollout
ROOT=Path(__file__).parent
starts=test_starts(7102,16)
heldout={k:cases(tracks(k,range(base,base+60)),starts) for k,base in [('similar',3000),('rough',4000)]}
rows=[]
for replicate in range(5):
    train_t,train_s=cases(tracks('similar',range(100+replicate*100,118+replicate*100)),train_starts(20261111+replicate,6))
    weights,curve=learn(train_t,train_s,'full',seed=41+replicate)
    for group,(T,S) in heldout.items():
        r=rollout(weights,T,S)
        rows.append((replicate,group,100+replicate*100,117+replicate*100,float(r['miss'].mean()),float(r['mse'].mean()),float(curve[-1][1])))
    print('Replicate',replicate,'similar',round(rows[-2][4]*100,2),'rough',round(rows[-1][4]*100,2))
with (ROOT/'Kinetra_habit_replications.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['replication','test_track_kind','first_training_track_seed','last_training_track_seed','mean_fraction_missed','mean_squared_error','final_train_objective']);w.writerows(rows)
# Confidence intervals resampling 60 independent test tracks, carrying 16 start conditions together
with (ROOT/'Kinetra_habit_per_track.csv').open(newline='') as f:
    records=list(csv.DictReader(f))
rng=np.random.default_rng(777)
ci=[]
for kind in ['similar','rough']:
    for comparator in ['passive','gap-only learned control','route replay','same corrections one gate late']:
        def track_rates(label):
            return np.array([float(r['fraction_gates_missed']) for r in records if r['track_kind']==kind and r['start_group']=='full_changed' and r['correction_policy']==label])
        before=track_rates(comparator); after=track_rates('habit learned (context)')
        assert len(before)==len(after)==60
        diff=before-after
        draws=rng.choice(diff, size=(10000,60),replace=True).mean(axis=1)
        ci.append((kind,comparator,float(diff.mean()),float(np.quantile(draws,.025)),float(np.quantile(draws,.975))))
with (ROOT/'Kinetra_habit_robustness.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['track_kind','comparison','absolute_reduction_missed_fraction','lower_ci95','upper_ci95']);w.writerows(ci)
print('Track-level paired differences (miss rate, percentage points) and bootstrap 95% interval')
for x in ci:print(x[0],x[1],tuple(round(float(y)*100,2) for y in x[2:]))
