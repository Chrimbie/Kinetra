#!/usr/bin/env python3
"""Kinetra: do retained *corrections* function as context-placed habits?
Reproducible randomized simulation; explicit ball dynamics, no target physics inversion.
Requires numpy. Usage: python Kinetra_habit_corrections.py
"""
from pathlib import Path
import csv, json, math
import numpy as np
from ball_track_transfer_test import make_track, GAP_HALF_WIDTH

HERE=Path(__file__).resolve().parent
N_STEPS=60
# Starting state (position, velocity, angle, spin)
ZERO=np.zeros((1,4))

def train_starts(seed, num=5):
    rng=np.random.default_rng(seed)
    s=np.column_stack([rng.uniform(-.6,.6,num),rng.uniform(-.5,.5,num),rng.uniform(-.85,.85,num),rng.uniform(-.85,.85,num)])
    s[0]=0
    return s

def test_starts(seed, num=16):
    rng=np.random.default_rng(seed)
    return np.column_stack([rng.uniform(-.9,.9,num),rng.uniform(-1,1,num),rng.uniform(-.9,.9,num),rng.uniform(-.9,.9,num)])

def tracks(kind, seeds):
    return np.array([[(g.center,g.push) for g in make_track(int(seed),kind)] for seed in seeds],dtype=np.float64)

def cases(t,starts):
    # Cross every ball start with each track; enforce separation of training and test cases
    return np.repeat(t,len(starts),axis=0),np.tile(starts,(len(t),1))

def rollout(w, T, S, mode='full', record=False, replay=None, delay_one=False):
    """Returns per-policy and per-episode scores. W shape (policies,10).
    Features are [opening - position, current velocity, current angle, current spin, constant].
    There are no previews of the next step's push. Angle and spin targets are bound to +/- .9.
    No weights change during rollout.
    """
    w=np.asarray(w,dtype=np.float64).reshape(-1,10)
    P,N=len(w),len(S)
    y=np.broadcast_to(S[None,:,0],(P,N)).copy()
    v=np.broadcast_to(S[None,:,1],(P,N)).copy()
    a=np.broadcast_to(S[None,:,2],(P,N)).copy()
    s=np.broadcast_to(S[None,:,3],(P,N)).copy()
    miss=np.zeros((P,N)); sq=np.zeros((P,N)); first=np.zeros((P,N))
    trace=[]
    prev_a=np.zeros((P,N)); prev_s=np.zeros((P,N))
    for i in range(T.shape[1]):
        center=T[:,i,0][None,:]
        push=T[:,i,1][None,:]
        e=center-y
        if replay is not None:
            aa,ss=replay[i]
            target_a=np.full((P,N),aa)
            target_s=np.full((P,N),ss)
        else:
            if mode=='full':
                vt,at,st=v,a,s
            elif mode=='gap_only':
                vt,at,st=0,0,0
            else: raise ValueError(mode)
            target_a=np.clip(w[:,0,None]*e + w[:,1,None]*vt + w[:,2,None]*at + w[:,3,None]*st +w[:,4,None],-.9,.9)
            target_s=np.clip(w[:,5,None]*e + w[:,6,None]*vt + w[:,7,None]*at + w[:,8,None]*st +w[:,9,None],-.9,.9)
        if delay_one:
            issued_a,issued_s=target_a,target_s
            target_a,target_s=prev_a,prev_s
            prev_a,prev_s=issued_a,issued_s
        a += .48*(target_a-a)
        s += .43*(target_s-s)
        v=.48*v+.83*np.sin(a)+.23*s+push
        y += v
        error=center-y
        missed=(np.abs(error)>GAP_HALF_WIDTH)
        miss+=missed
        sq+=error**2
        if i < 10: first+=missed
        if record:
            trace.append((i,center.copy(),push.copy(),y.copy(),v.copy(),a.copy(),s.copy(),target_a.copy(),target_s.copy(),error.copy()))
    return dict(miss=miss/N_STEPS,mse=sq/N_STEPS,first=first/10,trace=trace)

def learn(T,S,mode,seed=24,generations=55,pop_size=80):
    rng=np.random.default_rng(seed)
    mean=np.zeros(10)
    # Lower sigma in second half to refine action preferences.
    sigma=np.ones(10)*1.5
    bestw=np.zeros(10); bestloss=float('inf')
    curve=[]
    for gen in range(generations):
        w=rng.normal(mean,sigma,size=(pop_size,10))
        w=np.clip(w,-6,6)
        w[0]=bestw
        if gen==0: w[1]=0
        if mode=='gap_only': w[:,[1,2,3,6,7,8]]=0
        result=rollout(w,T,S,mode=mode)
        # Training objective uses all starts and all tracks. No eval sets seen.
        loss=np.mean(result['mse']+.12*result['miss'],axis=1)
        ix=np.argsort(loss)
        if loss[ix[0]] < bestloss:
            bestloss=loss[ix[0]]; bestw=w[ix[0]].copy()
        k=12
        mean=np.mean(w[ix[:k]],axis=0)
        sigma=0.83*sigma+0.17*np.std(w[ix[:k]],axis=0)
        sigma=np.maximum(sigma, .03)
        if gen>=20: sigma *= .976
        curve.append((gen+1,float(bestloss),float(np.mean(result['miss'][ix[0]])),float(sigma.mean())))
    return bestw,curve

def main():
    training_tracks=tracks('similar',range(100,118))
    training_starts=train_starts(20261111,6)
    train_t,train_s=cases(training_tracks,training_starts)
    print('Training via error-scored trial-and-error: 18 tracks x 6 start states; no upcoming push known to policy')
    print('Learning 10 context-sensitive correction weights, initially all zero.')
    full,curve=learn(train_t,train_s,'full',seed=41)
    print('Learning 4 context-blind weights, same budget.')
    simple,curvesimple=learn(train_t,train_s,'gap_only',seed=41)
    print('Full correction weights:',full.round(4).tolist())
    print('Simplified correction weights:',simple.round(4).tolist())
    print('Full learning objective start/end:',curve[0][1],curve[-1][1])
    print('Gap only learning objective start/end:',curvesimple[0][1],curvesimple[-1][1])
    # Same start+track+model must reproduce identical entire motion history.
    same_t,same_s=cases(tracks('similar',[3000]),np.array([[0,0,.4,.2]]))
    assert rollout(full,same_t,same_s,record=True)['trace'][0][3][0,0]==rollout(full,same_t,same_s,record=True)['trace'][0][3][0,0]
    path1=rollout(full,same_t,same_s,record=True)['trace']
    path2=rollout(full,same_t,same_s,record=True)['trace']
    assert all(np.array_equal(a[3],b[3]) and np.array_equal(a[5],b[5]) for a,b in zip(path1,path2))
    alt=rollout(full,same_t,np.array([[0,0,-.4,-.2]]),record=True)['trace']
    assert path1[0][3][0,0]!=alt[0][3][0,0]
    # Extract commands on training track to create fixed-index route replay; no retuning.
    source_t,source_s=cases(tracks('similar',[100]),np.zeros((1,4)))
    source=rollout(full,source_t,source_s,record=True)['trace']
    replay=np.array([(float(r[7][0,0]),float(r[8][0,0])) for r in source])
    # Held-out tracks and starts, distinct from train. 16 x 60 each = 960 episodes/class
    groups={
        'similar': tracks('similar',range(3000,3060)),
        'rough': tracks('rough',range(4000,4060)),
    }
    start_groups={
        'canonical':np.zeros((1,4)),
        'angle_spin_changed':np.column_stack([np.zeros(16),np.zeros(16),test_starts(7101,16)[:,2:]]),
        'full_changed':test_starts(7102,16),
    }
    policies=[('passive',np.zeros(10),'full',None),('habit learned (context)',full,'full',None),('habit no velocity/angle/spin',full,'gap_only',None),('gap-only learned control',simple,'gap_only',None),('route replay',full,'full',replay),('same corrections one gate late',full,'full','DELAY')]
    summary=[]; data=[]; examples={}
    for group,T in groups.items():
        for sg,S in start_groups.items():
            TT,SS=cases(T,S)
            runs={}
            for label,w,mode,seq in policies:
                r=rollout(w,TT,SS,mode=mode,replay=seq if isinstance(seq,np.ndarray) else None,delay_one=(seq=='DELAY' if isinstance(seq,str) else False))
                runs[label]=r
                miss=r['miss'].reshape(len(T),len(S))
                mse=r['mse'].reshape(len(T),len(S))
                first=r['first'].reshape(len(T),len(S))
                summary.append((group,sg,label,len(T),len(S),float(miss.mean()),float(mse.mean()),float(first.mean())))
                for j in range(len(T)):
                    data.append((group,sg,label,j,float(miss[j].mean()),float(mse[j].mean()),float(first[j].mean())))
            if group=='similar' and sg=='full_changed':
                examples['full_changed']=runs
    # One extra sensitivity check: compare final saved habit on identical novel tracks,
    # with only initial motion changed. Any result can be reproduced by exact replay.
    test_track=tracks('similar',[3000]); states=np.array([[0,0,.6,.4],[0,0,-.6,-.4],[0,0,0,0]])
    t2,s2=cases(test_track,states)
    trace=rollout(full,t2,s2,record=True)['trace']
    assert any(not np.allclose(line[7][0,0],line[7][0,1]) for line in trace)
    with (HERE/'Kinetra_habit_learning_curve.csv').open('w',newline='') as f:
        c=csv.writer(f);c.writerow(['generation','context_habit_best_loss','context_population_best_miss','context_sigma','gap_only_best_loss','gap_only_best_miss','gap_only_sigma'])
        c.writerows([a+b[1:] for a,b in zip(curve,curvesimple)])
    with (HERE/'Kinetra_habit_summary.csv').open('w',newline='') as f:
        c=csv.writer(f);c.writerow(['track_kind','start_group','correction_policy','number_tracks','starts_per_track','fraction_gates_missed','mean_squared_error','fraction_first_10_missed']);c.writerows(summary)
    with (HERE/'Kinetra_habit_per_track.csv').open('w',newline='') as f:
        c=csv.writer(f);c.writerow(['track_kind','start_group','correction_policy','test_track_index','fraction_gates_missed','mean_squared_error','fraction_first_10_missed']);c.writerows(data)
    with (HERE/'Kinetra_habit_example_trajectory.csv').open('w',newline='') as f:
        c=csv.writer(f);c.writerow(['initial_state_index','step','gate_center','push','y','velocity','angle','spin','commanded_angle','commanded_spin','residual'])
        for line in trace:
            for si in range(3):
                c.writerow((si,line[0],*[float(line[k][0,si]) for k in range(1,10)]))
    with (HERE/'Kinetra_habit_metadata.json').open('w') as f:
        json.dump(dict(training={'tracks':[100,117],'starts':training_starts.tolist(),'generations':55,'population':80,'seed':41,'objective':'mean of (MSE + 0.12 missed gate fraction)','optimizer':'cross entropy selection, no gradients and no target inverse-physics controller'},testing={'similar_seeds':[3000,3059],'rough_seeds':[4000,4059],'test_start_angle_spin_seed':7101,'test_start_full_seed':7102},learned_corrections=full.tolist(),gap_only_corrections=simple.tolist(),dynamics='identical to ball_track_transfer_test.py and kinetra_alignment_test.py',sensor_access='current opening center and current ball position/velocity/angle/spin; NO advance knowledge of obstacle push',limits='retained correction parameters updated by external trial-and-error optimizer, not spontaneous material change to the ball',determinism_checks_passed=True),f,indent=2)
    print('\nRESULTS (missed gates %, lower better)')
    for k,sg,label,nt,ns,mse,loss,early in summary:
        print(f'{k:7} {sg:19} {label:29} n={nt*ns:4d} missed={100*mse:6.2f}% early={100*early:6.2f}% MSE={loss:.4f}')
    print('Repeatability checks PASS; changed starting angle/spin changes trajectory and correction timing PASS')
    return summary
if __name__=='__main__':main()
