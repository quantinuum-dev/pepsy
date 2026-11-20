"""Extract Fig. 2 vector polylines from arXiv:2610.40776v1, not raw data."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def extract(pdf):
    from pypdf import PdfReader
    from pypdf.generic import ContentStream

    reader=PdfReader(pdf)
    form=reader.pages[2]['/Resources']['/XObject']['/Im2']
    if tuple(map(float,form['/BBox']))!=(0.,0.,231.463,178.):
        raise ValueError('Unexpected PDF figure layout')
    state=dict(matrix=np.eye(3),dash=[]);stack=[];points=[];curved=False;lines=[]
    for values,op in ContentStream(form,reader).operations:
        if op==b'q':stack.append(dict(state))
        elif op==b'Q':state=stack.pop()
        elif op==b'cm':
            a,b,c,d,e,f=map(float,values)
            state['matrix']=state['matrix']@np.array([[a,c,e],[b,d,f],[0,0,1]])
        elif op==b'd':state['dash']=list(values[0])
        elif op==b'm':
            points=[(state['matrix']@np.r_[list(map(float,values)),1])[:2]];curved=False
        elif op==b'l':points.append((state['matrix']@np.r_[list(map(float,values)),1])[:2])
        elif op==b'c':curved=True
        elif op in {b'S',b'B',b'f',b'n'}:
            if op==b'S' and len(points)>20 and not curved:
                p=np.asarray(points);panel='chain' if p[0,0]<100 else 'lattice'
                times=(p[:,0]-(35.3914 if panel=='chain' else 137.4219))/4.185
                values=(p[:,1]-27.4745)/4.07647
                lines.append(dict(panel=panel,reference=bool(state['dash']),
                                  times=times.tolist(),msd=values.tolist()))
            points=[]
    for panel,count in [('chain',5),('lattice',10)]:
        series=[line for line in lines if line['panel']==panel and not line['reference']]
        if len(series)!=count:raise ValueError('Unexpected number of paper curves')
        for width,line in zip(range(2,2*count+1,2),series):line['width']=width
    if len([line for line in lines if line['reference']])!=2:
        raise ValueError('Missing duplicate exact-reference curves')
    return dict(source='https://arxiv.org/pdf/2610.40776v1',
                pdf_sha256=hashlib.sha256(Path(pdf).read_bytes()).hexdigest(),
                kind='PDF vector coordinates, not authors raw simulation data',
                calibration='x ticks 0/10/20; y ticks 0/10/20/30',
                estimated_coordinate_rounding_msd=3e-4,
                typicality_uncertainty='not supplied; distinct from digitization error',curves=lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.write_text(json.dumps(extract(args.pdf),indent=2)+'\n')


if __name__=='__main__':main()
