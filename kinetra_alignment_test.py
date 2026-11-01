#!/usr/bin/env python3
"""Kinetra: test moment-to-moment physical alignment with no training or memory.

Reuses original Kinetra track generator, gate width and ball motion equations.
The fixed trajectory-alignment rule is analytically derived from these equations,
not fitted to any dataset; it does not change after any obstacle or trial.

Compare passive/no steering, direct center-pointing, trajectory-aware alignment
without access to the obstacle push, trajectory-aware alignment with perfect push
preview (an *oracle control*, not realistic), and the prior trained controller.

Nothing persists between runs; identical initial state and track replay exactly.
"""
import csv
import math
import random
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean, median
from ball_track_transfer_test import make_track, train, clip, GAP_HALF_WIDTH

OUT=Path(__file__).parent
DT_ANGLE=.48
DT_SPIN=.43

def controller(mode, gate, y, velocity, angle, spin, learned_weights=None):
    if mode == 'passive':
        return 0.,0.
    if mode == 'point_to_opening':
        return clip(gate.center-y, -.9,.9), 0.
    if mode == 'trained':
        p,d,wind,s,swind=learned_weights
        e=gate.center-y
        return clip(p*e-d*velocity-wind*gate.push,-.9,.9),clip(s*e-swind*gate.push,-.9,.9)
    if mode in ('align_now','align_oracle','align_wrong_start'):
        # Analytical inverse of the same one-step kinematic equations.
        # Goal: y_next = gate.center, using the ball's velocity/angle/spin.
        # Do not know obstacle's push unless ORACLE mode, so assume zero.
        observed_velocity=velocity
        observed_angle=angle
        observed_spin=spin
        if mode=='align_wrong_start':
            # Ablation: use current position, but incorrectly assume zero
            # current velocity, angle and spin on *every* step.
            observed_velocity=observed_angle=observed_spin=0.
        next_spin=(1.-DT_SPIN)*observed_spin
        assumed_push=gate.push if mode=='align_oracle' else 0.
        desired_dy=gate.center-y
        needed_sine=(desired_dy-.48*observed_velocity-.23*next_spin-assumed_push)/.83
        reachable=math.sin(.9)
        desired_angle=math.asin(clip(needed_sine,-reachable,reachable))
        angle_target=(desired_angle-(1.-DT_ANGLE)*observed_angle)/DT_ANGLE
        return clip(angle_target,-.9,.9),0.
    raise ValueError(mode)

def simulate(track, mode, start=(0.,0.,0.,0.), learned_weights=None, record=False):
    y,velocity,angle,spin=start
    misses=0
    squared_error=0.
    path=[]
    for index,gate in enumerate(track):
        aim_angle, aim_spin=controller(mode,gate,y,velocity,angle,spin,learned_weights)
        angle += DT_ANGLE*(aim_angle-angle)
        spin += DT_SPIN*(aim_spin-spin)
        velocity=.48*velocity+.83*math.sin(angle)+.23*spin+gate.push
        y += velocity
        err=gate.center-y
        misses += (abs(err)>GAP_HALF_WIDTH)
        squared_error += err*err
        if record:
            path.append((index,gate.center,gate.push,y,velocity,angle,spin,aim_angle,aim_spin,err))
    return (misses/len(track),squared_error/len(track),tuple(path))

def starts():
    rng=random.Random(20261101)
    return {
        'canonical':[(0.,0.,0.,0.)],
        'angle_spin_only':[(0.,0.,rng.uniform(-.4,.4),rng.uniform(-.4,.4)) for _ in range(40)],
        'full_moderate':[(rng.uniform(-.4,.4),rng.uniform(-.3,.3),rng.uniform(-.4,.4),rng.uniform(-.4,.4)) for _ in range(40)],
        'full_severe':[(rng.uniform(-.9,.9),rng.uniform(-1,1),rng.uniform(-.9,.9),rng.uniform(-.9,.9)) for _ in range(40)],
    }

def main():
    # Train on ONE old track, exclusively for trained comparison.
    learned,_=train(make_track(101),budget=4000,seed=12345)
    modes=('passive','point_to_opening','align_now','align_wrong_start','align_oracle','trained')
    groups={
        'new_similar_60':[make_track(1000+j) for j in range(60)],
        'new_rough_60':[make_track(2000+j,kind='rough') for j in range(60)]
    }
    start_groups=starts()
    # Assertions: identical state+track returns identical path; other start changes path.
    t=groups['new_similar_60'][0]
    zero=(0.,0.,0.,0.)
    trial=simulate(t,'align_now',zero,record=True)
    assert trial==simulate(t,'align_now',zero,record=True)
    assert trial[2] != simulate(t,'align_now',(0.,0.,.2,.1),record=True)[2]
    # No update/state between trials; replay is identical after running elsewhere.
    simulate(groups['new_rough_60'][0],'align_now',(.1,.1,-.2,.3))
    assert trial==simulate(t,'align_now',zero,record=True)

    all_rows=[]
    aggregate=defaultdict(list)
    for track_group,tracks in groups.items():
        for start_group,states in start_groups.items():
            for j,t in enumerate(tracks):
                for k,s in enumerate(states):
                    for mode in modes:
                        miss,mse,_=simulate(t,mode,s,learned)
                        all_rows.append((track_group,start_group,j,k,mode,*s,miss,mse))
                        aggregate[(track_group,start_group,mode)].append((miss,mse))
    summary=[]
    for key,v in sorted(aggregate.items()):
        miss=mean(x[0] for x in v)
        mse=mean(x[1] for x in v)
        med=median(x[0] for x in v)
        summary.append((*key,len(v),miss,mse,med))
    with (OUT/'kinetra_alignment_all_trials.csv').open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['track_group','start_group','track_index','start_index','mode','initial_position','initial_velocity','initial_angle','initial_spin','miss_fraction','mse'])
        w.writerows(all_rows)
    with (OUT/'kinetra_alignment_summary.csv').open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['track_group','start_group','mode','trials','mean_missed_fraction','mean_squared_error','median_missed_fraction'])
        w.writerows(summary)
    with (OUT/'kinetra_alignment_example_path.csv').open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['mode','step','gate_center','gate_push','ball_y','ball_velocity','ball_angle','ball_spin','command_angle','command_spin','signed_gate_error'])
        for mode in modes:
            result=simulate(t,mode,(0.,0.,.2,.1),learned,record=True)
            for row in result[2]:w.writerow((mode,*row))
    print('START STATE TEST: identical replay exactly; altered angle+spin gives different trajectory; no cross-run state: PASS')
    print('Trained comparator weights:',tuple(round(x,4) for x in learned))
    print('All no-training controllers fixed for all trials; no changes made from trials.')
    print('N trials:',len(all_rows))
    for track_group in groups:
        print('\n',track_group)
        for start_group in start_groups:
            print('  ',start_group)
            for mode in modes:
                v=aggregate[(track_group,start_group,mode)]
                print('   {:20s} {:6.2f}% missed MSE {:.4f}'.format(mode,100*mean(x[0] for x in v),mean(x[1] for x in v)))
    # Check cases where same starts lead to different performance based on start
    for group,tracks in groups.items():
        vals=aggregate[(group,'full_severe','align_now')]
        print('Severe-start alignment',group,'miss fraction min/max:',round(min(x[0] for x in vals),3),round(max(x[0] for x in vals),3))
    # Generate concise machine-readable summary.
    (OUT/'kinetra_alignment_metadata.json').write_text(json.dumps({
        'physical_equations':{
            'angle_next':'angle + 0.48*(target_angle-angle)',
            'spin_next':'spin + 0.43*(target_spin-spin)',
            'velocity_next':'0.48*velocity + 0.83*sin(angle_next) + 0.23*spin_next + gate.push',
            'position_next':'position + velocity_next',
        },
        'gate_half_width':GAP_HALF_WIDTH,
        'start_ranges':{'angle_spin_only':'initial y and v=0; angle/spin each [-0.4,0.4]',
                        'full_moderate':'y angle spin each [-0.4,0.4]; v [-0.3,0.3]',
                        'full_severe':'y angle spin each [-0.9,0.9]; v [-1,1]'},
        'number_of_trials':len(all_rows),
        'trained_weights_for_comparator':learned,
        'no_train_alignment':True,
        'no_persistent_state':True,
        'main_alignment_observes':'gate.center and current ball state; does not preview gate.push',
        'oracle_alignment_observes':'gate.center, current ball state and gate.push',
        'random_seed_for_start_states':20261101,
    },indent=2))

if __name__=='__main__': main()
