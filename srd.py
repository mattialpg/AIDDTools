import pandas
import math
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

def crrn(n, res=1000):
    ''' Calculates CRRN without ties(!).
    Needs the following input:
    - n: number of rows, practically given by len(df)
    - res: resolution, or number of points on the normal distribution (ignored for n<7)
    '''
    
    [XX1,Med,XX19]=[None,None,None]
    
    if n<2:
        raise CRRNError('Smallest number of objects is 2.')
    elif n==2:
        x=[0,50]
        y=[50,50]
    elif n==3:
        x=[0,50,100]
        y=[16.67,33.33,50]
    elif n==4:
        x=[0,25,50,75,100]
        y=[4.17,12.50,29.17,37.50,16.67]
    elif n==5:
        x=[0,16.67,33.3,50,66.67,83.33,100]
        y=[0.83,3.33,10.00,20.00,29.17,20.00,16.67]
    elif n==6:
        x=[0,11.11,22.22,33.33,44.44,55.56,66.67,77.78,88.89,100]
        y=[0.14,0.69,2.50,6.39,12.92,19.03,20.56,18.89,13.89,5.00]
        
    elif n<16:
        ''' Cumulative frequencies.'''
        mt = {7:62.74,8:62.74,9:64.40,10:64.21,11:65.19,12:64.99,13:65.64,14:65.48,15:65.87}[n]
        st = {7:21.62,8:19.46,9:18.36,10:17.02,11:16.23,12:15.30,13:14.70,14:14.03,15:13.59}[n]
        
        x = np.linspace(0.0, 100.0, res)
        y = [(math.tanh( (i-mt)/st ) - math.tanh( (0-mt)/st ) ) / 2 for i in x]
        
        XX1 = next(i for (i,j) in zip(x,y) if j > 0.05)
        Med = next(i for (i,j) in zip(x,y) if j > 0.5)
        XX19 = next(i for (i,j) in zip(x,y) if j > 0.95)
        
    elif n<30:
        ''' This might correspond to the ClassTH=1 case of the with-ties distributions!'''
        mean={16:66.713,17:67.021,18:66.734,19:66.952,20:66.721,21:66.913,22:66.786,23:66.908,24:66.714,
              25:66.883,26:66.800,27:66.856,28:66.834,29:66.845}[n]
        std={16:11.049,17:10.842,18:10.392,19:10.153,20:9.871,21:9.596,22:9.349,23:9.163,24:8.854,
             25:8.722,26:8.561,27:8.409,28:8.247,29:8.074}[n]

        x = np.linspace(0.0, 100.0, res)
        y = scipy.stats.norm.pdf(x,mean,std)
        
        cumFreq = scipy.stats.norm.cdf(x,mean,std)
        
        XX1 = next(i for (i,j) in zip(x,cumFreq) if j > 0.05)
        Med = next(i for (i,j) in zip(x,cumFreq) if j > 0.5)
        XX19 = next(i for (i,j) in zip(x,cumFreq) if j > 0.95)
        
    else:
        [a,b] = [2.3796,-0.3509]

        mean = 66.667
        std = 100 / ( b + a*math.sqrt(n) )
        
        x = np.linspace(0.0, 100.0, res)
        y = scipy.stats.norm.pdf(x,mean,std)
        
        cumFreq = scipy.stats.norm.cdf(x,mean,std)
        
        XX1 = next(i for (i,j) in zip(x,cumFreq) if j > 0.05)
        Med = next(i for (i,j) in zip(x,cumFreq) if j > 0.5)
        XX19 = next(i for (i,j) in zip(x,cumFreq) if j > 0.95)
        
    return [x,y,XX1,Med,XX19]

def srd_core(df,ref,normalize=True):
    ''' Shortcut for the core SRD calculation.'''
    
    refVector=calc_ref(df,ref)
    
    dfr=df.rank()
    rVr=refVector.rank()
    diffs=dfr.subtract(rVr,axis=0)

    srd_values=diffs.abs().sum()
    
    if normalize == True:
        k = math.floor(len(df)/2)
        if len(df)%2 == 0: maxSRD = 2 * k**2
        else: maxSRD = 2 * k * (k+1)
            
        srd_values=srd_values/maxSRD*100
    
    return srd_values

def calc_ref(df,ref,axis=1):
    '''Select reference column or produce one with a data fusion method.'''
    if axis==1:
        if ref in df.columns:
            refVector=df[ref]
        elif ref in ['min','max','mean','median']:
            refVector={
                'min': df.min(axis=axis),
                'max': df.max(axis=axis),
                'mean': df.mean(axis=axis),
                'median': df.median(axis=axis),
                        }[ref]
        else:
            raise ReferenceError('Column not found.')
    
    '''Supports row-wise reference selection/calculation, but this is not yet implemented in srd_core!'''
    if axis==0:
        if ref in df.index:
            refVector=df.loc[ref]
        elif ref in ['min','max','mean','median']:
            refVector={
                'min': df.min(axis=axis),
                'max': df.max(axis=axis),
                'mean': df.mean(axis=axis),
                'median': df.median(axis=axis),
                        }[ref]
        else:
            raise ReferenceError('Row not found.')
    
    return refVector
    
def calc_RSD(V, ref_method='mean', rank_method='min'):
    # Calculate reference vector and ranks
    rV = V.rank(method = rank_method)
    R = calc_ref(rV, ref_method)
    rR = R.rank()

    # Calculate rank differences and their sum
    RD = rV.subtract(rR, axis=0)
    SRD = RD.abs().sum()

    # Find maximum SRD value and normalize
    k = math.floor(len(V)/2)
    if len(V)%2 == 0: SRD_max = 2*k**2
    else: SRD_max = 2*k*(k+1)
    SRD_nor = 100*SRD/SRD_max
    
    return SRD, SRD_nor

def cross_val(V):
    from sklearn.model_selection import LeaveOneOut, KFold

    if len(V) < 14: cv_iterator = LeaveOneOut()       # Leave-one-out cross-validation. Recommended for <14 samples
    else: cv_iterator = KFold(n_splits=7)             # N-fold cross-validation. Recommended for >13 samples

    srd_collector=[]
    for train_index, test_index in cv_iterator.split(V):
        srd_current = srd.srd_core(V.iloc[train_index],'mean')
        srd_collector.append(srd_current)

    return pd.DataFrame(srd_collector)

def SRD_plot(SRD_nor, save=None):
    """x value and column height means distance from the reference: the smaller the better"""
    my_colors = ['red','blue','green','purple','orange','brown','magenta','teal','violet','lime',
                'darkorange', 'darkorchid', 'darkred', 'darkseagreen', 'darkslateblue',
                 'darkslategrey', 'darkturquoise', 'darkviolet','indianred', 'indigo','black']

    fig, ax = plt.subplots()
    bars = ax.bar(SRD_nor, SRD_nor, width=0.2, color=my_colors)
    ax.set_title('SRD results')
    ax.set_xlabel('SRD (%)'); ax.set_ylabel('SRD (%)')
    ax.set_xlim(0,100); ax.set_ylim(0,100)

    texts = []
    for j,rect in enumerate(bars):
        left = rect.get_x()+1
        top = rect.get_y()+rect.get_height()
        texts.append(ax.text(left, top, SRD_nor.index[j],
                             color = rect.get_facecolor(), fontweight='semibold'))

    # This module is highly recommended for automatically adjusting the text positions
    # on the SRD plot - so that it is less cluttered. Can be installed from: https://github.com/Phlya/adjustText
    try:
        from adjustText import adjust_text
        adjust_text(texts, add_objects=bars, autoalign=True, force_objects=(0.1,1.0), 
                    only_move={'points':'xy', 'text':'y', 'objects':'xy'}, ha='center', va='bottom')
    except: print('Cannot adjust text'); pass

# # Calculate distribution of random SRD values and significant points
    # [x, y, XX1, Med, XX19] = crrn(len(SRD_nor))

    # if XX1:
        # ax.vlines(XX1, 0, 100, label='XX1', color='grey', linestyle='dashed')
        # ax.annotate('XX1', xy=(XX1, 100), xytext=(5, -10), textcoords="offset points")
    # if Med:
        # ax.vlines(Med, 0, 100, label='Med', color='grey', linestyle='dashed')
        # ax.annotate('Med', xy=(Med, 100), xytext=(5, -10), textcoords="offset points")
    # if XX19:
        # ax.vlines(XX19, 0, 100, label='XX19', color='grey', linestyle='dashed')
        # ax.annotate('XX19', xy=(XX19, 100), xytext=(5, -10), textcoords="offset points")
    
    # # s = np.random.normal(mu, sigma, 1000)

    # ax2 = ax.twinx()
    # ax2.plot(x,y,color='black')
    # ax2.set_ylabel('Rel. frequencies of SRD')
    # ax2.set_ylim(bottom=0)


    fig.tight_layout()
    # plt.savefig('srd_plot.png',dpi=300,bbox_inches='tight')
    plt.show()