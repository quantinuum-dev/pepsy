"""Figure 2 physics from arXiv:2610.40776 with Pepsy's protected-SVD TEBD.

Open length-21, uncoupled mixed-field Ising chains are interleaved column by
column. This benchmark matches the Hamiltonian and measured correlator, but
uses Strang transfers rather than the paper's TDVP integrator.
"""

import argparse
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np

from pepsy.backends import TorchLinalgConfig
from pepsy.operators import PauliMPO
from pepsy.optimizers import PauliDMT

try:
    from examples.dmt_2d_window import field_transfer, zz_transfer
except ModuleNotFoundError:
    from dmt_2d_window import field_transfer, zz_transfer


GX, GZ = 1.4, .9045


def problem(length=21, width=2):
    """Paper's column-major interleaving j=x*width+y; eta=0."""
    if length < 3 or length % 2 != 1 or width < 1:
        raise ValueError('Require odd length >=3 and positive width')
    n = length*width
    edges = [(x*width+y, (x+1)*width+y) for x in range(length-1) for y in range(width)]
    neighbors = [set() for _ in range(n)]
    for x in range(length):
        for y in range(width):
            j = x*width+y
            for a,b in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                if 0 <= a < length and 0 <= b < width:
                    neighbors[j].add(a*width+b)
    triples = {tuple(sorted((j,a,b))) for j, near in enumerate(neighbors)
               for a,b in combinations(sorted(near),2)}
    columns = []
    for x in range(length):
        terms = []
        for y in range(width):
            j=x*width+y
            terms.extend([(GX,(j,),'X'),(GZ,(j,),'Z')])
            if x:
                terms.append((.5,(j-width,j),'ZZ'))
            if x+1 < length:
                terms.append((.5,(j,j+width),'ZZ'))
        columns.append(terms)
    return dict(length=length,width=width,n=n,edges=edges,regions=tuple(sorted(triples)),
                columns=columns,initial_terms=columns[length//2],
                normalization=width*(GX**2+GZ**2+1.))


def initial_operator(model, dtype='float64'):
    """Exact sparse energy packet, converted locally to native Torch cores."""
    import torch

    op=PauliMPO.from_terms(model['n'],model['initial_terms'])
    cores=[torch.as_tensor(a.real if dtype=='float64' else a, dtype=getattr(torch,dtype))
           for a in op.to_pauli_cores()]
    return PauliMPO.from_pauli_cores(cores)


def coefficients(cores, words):
    """Independent Pauli-coordinate contractions, without a 2**N trace."""
    n=len(cores)
    left=[np.ones(1)]
    right=[None]*(n+1);right[n]=np.ones(1)
    for a in cores:
        left.append(left[-1]@a[:,:,0])
    for j in range(n-1,-1,-1):
        right[j]=cores[j][:,:,0]@right[j+1]
    values=[]
    labels={'I':0,'X':1,'Y':2,'Z':3}
    for sites,word in words:
        choices=dict(zip(sites,word));a,b=min(sites),max(sites)
        panel=left[a]
        for j in range(a,b+1):
            panel=panel@cores[j][:,:,labels[choices.get(j,'I')]]
        values.append(panel@right[b+1])
    return np.asarray(values),left[-1].item()


def measure(run,model):
    cores=[a.detach().cpu().numpy() for a in run.state.to_pauli_cores()]
    terms=[term for column in model['columns'] for term in column]
    values,identity=coefficients(cores,[(sites,word) for _,sites,word in terms])
    cx=[];offset=0
    for column in model['columns']:
        cx.append(sum(c*values[offset+k] for k,(c,_,_) in enumerate(column)))
        offset+=len(column)
    cx=np.asarray(cx)/model['normalization']
    x=np.arange(model['length'])-model['length']//2
    msd=np.sum(cx*x*x)-np.sum(cx*x)**2
    return dict(msd=float(msd.real),correlator=cx.real.tolist(),
                total_weight=float(cx.real.sum()),mean=float((cx*x).real.sum()),
                max_imaginary_correlator=float(abs(cx.imag).max()),
                identity_coefficient=float(abs(identity)),
                actual_max_bond=max(a.shape[1] for a in cores))


def run_benchmark(*,length=21,width=2,chi=128,dt=.2,time=20.,protection='lattice',
                  dtype='float64',cpu_svd='scipy_gesdd',max_seconds=900.,output=None,
                  check_regions=False):
    import torch

    steps=round(time/dt)
    if dt<=0 or abs(steps*dt-time)>1e-9 or protection not in {'lattice','chain'}:
        raise ValueError('Require an integral number of timesteps and known protection')
    torch.set_num_threads(1)
    model=problem(length,width)
    policy=TorchLinalgConfig(cpu_svd=cpu_svd,cpu_qr='scipy_pivoted',quimb_split_drivers=True)
    result=dict(length=length,width=width,chi=chi,dt=dt,requested_time=time,
                protection=protection,dtype=dtype,backend='torch',device='cpu',
                cpu_svd=cpu_svd,cpu_qr='scipy_pivoted',integrator='Strang field/ZZ/field',
                fields=dict(gx=GX,gz=GZ,eta=0.),site_order='column-major: x*width+y',
                status='running',completed_steps=0,samples=[],step_seconds=[],
                preservation_errors=[],trace_guard=False,check_regions=check_regions)

    def checkpoint():
        if output:
            Path(output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

    with policy.activated():
        run=PauliDMT(initial_operator(model,dtype),max_bond=chi,variant='operator',
                     mode='heisenberg',trace_guard=False,cutoff=0.,
                     regions=model['regions'] if protection=='lattice' else None,
                     region_selection='min_cover',check_regions=check_regions,
                     preservation_check='warn')
        native=lambda a:torch.as_tensor(a,dtype=getattr(torch,dtype))
        field=(native(field_transfer(-dt,GX,GZ)).reshape(1,1,4,4),)
        gates=tuple(native(a) for a in zz_transfer(width+1,-dt))
        result['samples'].append(dict(time=0.,**measure(run,model)))
        checkpoint()
        sample_every=max(1,round(1./dt))
        try:
            for step in range(1,steps+1):
                started=perf_counter()
                # Alternate complete sweeps to carry the canonical centre.
                # This order is a declared finite-chi schedule choice.
                forward=step%2==1
                sites=range(model['n']) if forward else range(model['n']-1,-1,-1)
                edges=model['edges'] if forward else reversed(model['edges'])
                absorb='right' if forward else 'left'
                for j in sites:
                    report=run.apply_transfer_mpo(j,field,absorb=absorb)
                    result['preservation_errors'].extend(report.preservation_errors)
                for a,b in edges:
                    report=run.apply_transfer_mpo(a,gates,absorb=absorb)
                    result['preservation_errors'].extend(report.preservation_errors)
                for j in sites:
                    report=run.apply_transfer_mpo(j,field,absorb=absorb)
                    result['preservation_errors'].extend(report.preservation_errors)
                result['step_seconds'].append(perf_counter()-started)
                result['completed_steps']=step
                exhausted=sum(result['step_seconds'])>=max_seconds
                if step%sample_every==0 or step==steps or exhausted:
                    sample=dict(time=step*dt,**measure(run,model))
                    result['samples'].append(sample)
                    print(f"{protection} {length}x{width} chi{chi} dt{dt} "
                          f"t={step*dt:.2f} MSD={sample['msd']:.6g} "
                          f"step={result['step_seconds'][-1]:.3g}s",flush=True)
                checkpoint()
                if exhausted:
                    break
            result['status']='completed' if result['completed_steps']==steps else 'time_budget'
        except (ValueError,RuntimeError,FloatingPointError) as exc:
            result.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        checkpoint()
    return result


def exact_reference(length,times):
    """Small-chain exact spectral calculation, independent of DMT/transfer code."""
    if length>10:
        raise ValueError('Dense reference is restricted to at most 10 spins')
    from scipy.linalg import eigh

    model=problem(length,1);d=2**length
    paulis={'I':np.eye(2),'X':np.array([[0,1],[1,0]]),'Z':np.diag([1,-1])}
    def matrix(terms):
        out=np.zeros((d,d))
        for c,sites,word in terms:
            choices=dict(zip(sites,word));a=np.ones((1,1))
            for j in range(length):a=np.kron(a,paulis[choices.get(j,'I')])
            out+=c*a
        return out
    columns=[matrix(terms) for terms in model['columns']]
    e,v=eigh(sum(columns));h0=v.T@columns[length//2]@v
    probes=np.stack([v.T@a@v for a in columns])
    x=np.arange(length)-length//2
    result=[]
    for t in times:
        evolved=np.exp(1j*t*(e[:,None]-e[None,:]))*h0
        cx=np.einsum('xab,ba->x',probes,evolved).real/d/model['normalization']
        result.append(dict(time=float(t),correlator=cx.tolist(),msd=float(cx@(x*x)-(cx@x)**2),total_weight=float(cx.sum())))
    return result


def sparse_chain(length):
    """Independent bit-basis Hamiltonian; memory O(length*2**length)."""
    from scipy.sparse import csr_matrix

    d=2**length;indices=np.arange(d,dtype=np.int32)
    diagonal=np.zeros(d)
    previous=None
    for j in range(length):
        spin=1.-2.*((indices>>(length-1-j))&1)
        diagonal+=GZ*spin
        if previous is not None:diagonal+=previous*spin
        previous=spin
    columns=np.empty((d,length+1),dtype=np.int32)
    data=np.full((d,length+1),GX)
    for j in range(length):columns[:,j]=indices^(1<<(length-1-j))
    columns[:,-1]=indices;data[:,-1]=diagonal
    return csr_matrix((data.ravel(),columns.ravel(),
                       np.arange(0,d*(length+1)+1,length+1,dtype=np.int32)),shape=(d,d))


def energy_action(vector,column,length):
    """Apply a centered local energy density independently of MPO code."""
    indices=np.arange(2**length,dtype=np.int32)
    spin=1.-2.*((indices>>(length-1-column))&1)
    diagonal=GZ*spin
    for neighbor in (column-1,column+1):
        if 0<=neighbor<length:
            diagonal+=.5*spin*(1.-2.*((indices>>(length-1-neighbor))&1))
    return diagonal*vector+GX*vector[indices^(1<<(length-1-column))]


def typicality_reference(length=21,time=20,seeds=(712,),output=None):
    """Independent continuous-time sparse exponential, stochastic T=infinity trace.

    Keeps at most six pairs of 2**length vectors at once, never a density
    matrix. One seed has no statistical error estimate; it is not exact data.
    """
    from scipy.sparse.linalg import expm_multiply

    if int(time)!=time or time<1 or not seeds:
        raise ValueError('Reference uses positive integer final time and at least one seed')
    started=perf_counter();h=-1j*sparse_chain(length)
    norm=GX**2+GZ**2+1.;times=np.arange(int(time)+1);x=np.arange(length)-length//2
    result=dict(length=length,method='continuous-time sparse exponential + random-phase typicality',
                seeds=list(seeds),samples_by_seed=[],status='running',statistical_error='unestimated for one seed')
    def save():
        if output:Path(output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for seed in seeds:
        rng=np.random.default_rng(seed)
        random=np.exp(2j*np.pi*rng.random(2**length))/np.sqrt(2**length)
        state=np.column_stack((random,energy_action(random,length//2,length)))
        rows=[]
        def sample(t,pair):
            cx=np.array([np.vdot(pair[:,0],energy_action(pair[:,1],j,length)).real
                         for j in range(length)])/norm
            rows.append(dict(time=float(t),correlator=cx.tolist(),msd=float(cx@(x*x)-(cx@x)**2),
                             total_weight=float(cx.sum())))
        sample(0,state)
        for start in range(0,int(time),5):
            count=min(5,int(time)-start)
            states=expm_multiply(h,state,start=0.,stop=float(count),num=count+1,traceA=0.)
            for k in range(1,count+1):sample(start+k,states[k])
            state=states[-1].copy();del states
            result['samples_by_seed']=result['samples_by_seed'][:list(seeds).index(seed)]+[rows]
            result['seconds']=perf_counter()-started;save()
            print(f'typicality N={length} seed={seed} t={start+count} '
                  f'MSD={rows[-1]["msd"]:.6g} seconds={result["seconds"]:.1f}',flush=True)
        del state
    curves=np.array([[row['correlator'] for row in rows] for rows in result['samples_by_seed']])
    mean=curves.mean(axis=0)
    result['samples']=[dict(time=float(t),correlator=cx.tolist(),msd=float(cx@(x*x)-(cx@x)**2),
                            total_weight=float(cx.sum())) for t,cx in zip(times,mean)]
    if len(seeds)>1:
        result['statistical_error']='standard error across independent random-phase states'
        result['msd_standard_error']=(np.std([[r['msd'] for r in rows] for rows in result['samples_by_seed']],
                                            axis=0,ddof=1)/np.sqrt(len(seeds))).tolist()
    result.update(status='completed',seconds=perf_counter()-started);save()
    return result


def summarize(paths,reference,output):
    """Save matched physics comparisons and plots, retaining incomplete runs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    paper=json.loads((Path(__file__).parent/'data/lattice_dmt_fig2.json').read_text())
    exact=next(c for c in paper['curves'] if c['reference'])
    oracle=json.loads(Path(reference).read_text()) if reference else None
    rows=[json.loads(Path(p).read_text())|{'source':Path(p).name} for p in paths]
    if any(row['length']!=21 for row in rows) or (oracle and oracle['length']!=21):
        raise ValueError('The published Fig. 2 comparison requires length 21')
    for row in rows:
        published=next((c for c in paper['curves'] if not c['reference']
                        and c['panel']==row['protection'] and c['width']==row['width']),None)
        for s in row['samples']:
            s['paper_reference_msd']=float(np.interp(s['time'],exact['times'],exact['msd']))
            if published:
                s['paper_matching_method_msd']=float(np.interp(s['time'],published['times'],published['msd']))
            if oracle and 'samples' in oracle:
                ts=[r['time'] for r in oracle['samples']]
                if ts[0]<=s['time']<=ts[-1]:
                    s['typicality_reference_msd']=float(np.interp(s['time'],ts,[r['msd'] for r in oracle['samples']]))
        row['max_total_weight_defect']=max(abs(s['total_weight']-1) for s in row['samples'])
        if published:
            row['max_msd_difference_from_paper_method']=max(abs(s['msd']-s['paper_matching_method_msd'])
                                                           for s in row['samples'])
    result=dict(paper=paper,reference=oracle,runs=rows,
                comparison='Same Hamiltonian, energy packet and geometry; protected-SVD TEBD rather than TDVP',
                timing='Concurrent CPU jobs; not an isolated speed benchmark')
    prefix=Path(output)
    prefix.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,panel in zip(axes[0],['chain','lattice']):
        first=True
        for c in paper['curves']:
            if c['panel']==panel and not c['reference']:
                ax.plot(c['times'],c['msd'],color='0.7',lw=1,alpha=.7,
                        label='Paper DMT curves (digitized)' if first else '_nolegend_')
                first=False
        ax.plot(exact['times'],exact['msd'],'k--',label='Paper reference (digitized)')
        if oracle and 'samples' in oracle:
            ax.plot([r['time'] for r in oracle['samples']],[r['msd'] for r in oracle['samples']],
                    color='black',lw=1,label=f"Independent typicality ({len(oracle['seeds'])} seed)")
        ax.set(title=f'{panel.capitalize()} DMT (paper χ=128)',xlabel='Time',ylabel='Energy MSD',xlim=(0,20),ylim=(0,35))
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    for k,row in enumerate(rows):
        if row['status'] not in {'completed','time_budget'} or row['preservation_errors']:continue
        t=[r['time'] for r in row['samples']];msd=[r['msd'] for r in row['samples']]
        color=colors[k%len(colors)]
        label=f"{row['protection']} W={row['width']}, χ={row['chi']}, dt={row['dt']}"
        if row['status']=='time_budget':label+=f" (partial, t≤{t[-1]:g})"
        panel=0 if row['protection']=='chain' else 1
        axes[0,panel].plot(t,msd,'o-',ms=3,color=color,label=label)
        axes[1,0].plot(t,[abs(r['msd']-r.get('typicality_reference_msd',r['paper_reference_msd']))
                          for r in row['samples']],color=color,label=label)
        axes[1,1].semilogy(t,np.maximum([abs(r['total_weight']-1) for r in row['samples']],1e-16),
                           color=color,label=label)
    label=(f"Difference from {len(oracle['seeds'])}-seed typicality" if oracle and 'samples' in oracle
           else 'Difference from paper reference')
    axes[1,0].set(title=label,xlabel='Time',ylabel='Absolute MSD difference')
    axes[1,1].set(title='Energy-overlap conservation',xlabel='Time',ylabel='|sum normalized C − 1|')
    for ax in axes.ravel():ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(prefix.with_suffix('.png'),dpi=160)
    fig.savefig(prefix.with_suffix('.svg'))
    svg=prefix.with_suffix('.svg');svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--length',type=int,default=21)
    parser.add_argument('--width',type=int,default=2)
    parser.add_argument('--chi',type=int,default=128)
    parser.add_argument('--dt',type=float,default=.2)
    parser.add_argument('--time',type=float,default=20.)
    parser.add_argument('--protection',choices=['lattice','chain'],default='lattice')
    parser.add_argument('--dtype',choices=['float64','complex128'],default='float64')
    parser.add_argument('--cpu-svd',choices=['scipy_gesdd','scipy_gesvd','torch'],default='scipy_gesdd')
    parser.add_argument('--max-seconds',type=float,default=900.)
    parser.add_argument('--check-regions',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('/tmp/dmt-lattice-fig2.json'))
    parser.add_argument('--typicality',action='store_true',help='Independent single-chain sparse reference')
    parser.add_argument('--seeds',nargs='+',type=int,default=[712])
    parser.add_argument('--summarize',nargs='+',type=Path,help='Aggregate saved runs and plot; skip evolution')
    parser.add_argument('--reference',type=Path,help='Independent typicality JSON for summary')
    args=vars(parser.parse_args());typicality=args.pop('typicality');seeds=args.pop('seeds')
    paths=args.pop('summarize');reference=args.pop('reference')
    if paths:
        summarize(paths,reference,args['output'])
    elif typicality:
        typicality_reference(args['length'],args['time'],seeds,args['output'])
    else:
        run_benchmark(**args)


if __name__=='__main__':main()
