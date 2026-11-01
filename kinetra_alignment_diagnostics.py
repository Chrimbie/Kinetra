#!/usr/bin/env python3
"""Diagnostic: starting-state dependence and recovery of alignment, no tuning."""
import csv
from pathlib import Path
from statistics import mean
from kinetra_alignment_test import simulate, starts
from ball_track_transfer_test import make_track, GAP_HALF_WIDTH

OUT=Path(__file__).parent
tracks={'similar':[make_track(1000+i) for i in range(60)],
        'rough':[make_track(2000+i,kind='rough') for i in range(60)]}
st=starts()

def errfail(row):
    return abs(row[-1])>GAP_HALF_WIDTH

aggregates=[]
for kind,track_list in tracks.items():
    for category in ('angle_spin_only','full_moderate','full_severe'):
        stats={'first_miss':[],'first5_miss':[],'last5_miss':[],
               'diff_pos_1':[],'diff_pos_2':[],'diff_pos_5':[],'diff_pos_10':[],'diff_pos_20':[],'diff_pos_60':[]}
        for track in track_list:
            can=simulate(track,'align_now',start=(0.,0.,0.,0.),record=True)[2]
            for s in st[category]:
                path=simulate(track,'align_now',start=s,record=True)[2]
                stats['first_miss'].append(int(errfail(path[0])))
                stats['first5_miss'].append(mean(errfail(x) for x in path[:5]))
                stats['last5_miss'].append(mean(errfail(x) for x in path[-5:]))
                for step in [1,2,5,10,20,60]:
                    stats[f'diff_pos_{step}'].append(abs(path[step-1][3]-can[step-1][3]))
        aggregates.append({'track_group':kind,'start_group':category,
                           'n':len(stats['first_miss']),
                           **{k:mean(v) for k,v in stats.items()}})
with (OUT/'kinetra_alignment_diagnostics.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=aggregates[0].keys());writer.writeheader();writer.writerows(aggregates)
for r in aggregates:
    print('{track_group:7s} {start_group:18s} n={n}, first gate missed={first_miss:.2%}, first 5 missed={first5_miss:.2%}, last 5 missed={last5_miss:.2%}, avg displacement vs canonical at gate 1={diff_pos_1:.4f}, 10={diff_pos_10:.4f}, 60={diff_pos_60:.4f}'.format(**r))
