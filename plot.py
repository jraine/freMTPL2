import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import numpy as np
import pathlib

matplotlib.use('agg')

def plot_overlay(sample, vars=None, 
                 path=None, name=None, logscale=False):

    if vars is None:
        vars = [f'Feat. {i}' for i in range(sample.shape[1])]
    elif type(vars) is str:
        vars = [vars]
    
    fig,axes = plt.subplots(nrows=1,ncols=len(vars),figsize=(7*len(vars),5))
    if sample.shape[1]==1:
        axes = [axes]
    for i,(ax,lab) in enumerate(zip(axes,vars)):
        _,bins,_ = ax.hist(sample[:,i],weights=np.ones(len(sample))/len(sample),bins=50,alpha=0.5)
        ax.set_xlabel(lab)
        ax.semilogy(logscale)

    if path is not None:
        path = pathlib.Path(path)
        path.mkdir(parents=True,exist_ok=True)
        fig.tight_layout()
        fig.savefig(path/ f'{name}.pdf')

def plot_categorical(sample, vars=None, 
                 path=None, name=None):

    if vars is None:
        vars = [f'Feat. {i}' for i in range(sample.shape[1])]
    elif type(vars) is str:
        vars = [vars]
    
    fig,axes = plt.subplots(nrows=1,ncols=len(vars),figsize=(7*len(vars),5))
    if sample.shape[1]==1:
        axes = [axes]
    for i,(ax,lab) in enumerate(zip(axes,vars)):
        _,bins,_ = ax.hist(np.argmax(sample[:,i],axis=1) ,weights=np.ones(len(sample))/len(sample),bins=50,alpha=0.5)
        ax.set_xlabel(lab)

    if path is not None:
        path = pathlib.Path(path)
        path.mkdir(parents=True,exist_ok=True)
        fig.tight_layout()
        fig.savefig(path/ f'{name}.pdf')