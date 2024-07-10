#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 14:20:55 2022
Code to accompany the paper 'Animal pilot studies should not be used to estimate sample size if effect size and population variance are unknown' (Bird, Jedlicka, & Wilhelm)
@author: Alexander D Bird
"""
import scipy.stats as st 
from scipy.special import beta , factorial 
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np

colours_A=[
    '#003f5c',
    '#58508d',
    '#bc5090',
    '#ff6361',
    '#ffa600',
    ]

colours_C=[
    '#003f5c',
    '#444e86',
    '#955196',
    '#dd5182',
    '#ff6e54',
    '#ffa600'
    ]

def density_invncf(x,d,s,psz):
    lam=(psz*(d**2))/(s**2)
    pref=x**((psz-3)/2)/(np.exp(lam/2)*(16**((psz-1)/2)))
    run_sum=0
    for ward in range(0,100):
        summand=((lam/2)**(ward))*((1+x/16)**(-ward-psz/2))/(factorial(ward)*beta((psz-1)/2,ward+1/2))
        run_sum+=summand
    return pref*run_sum

def explicit_pdf(d,s,psz,res=1000,x_max=100):
    x =np.linspace(0,x_max,res)
    y = np.zeros(res)
    for ward in range(res):
        y[ward]=density_invncf(x[ward],d,s,psz)
    return x , y


def stable_pdf(d,s,psz,res=100,x_max=100):
    rv = st.ncf(1,psz-1,psz*d**2/(s**2)) # Initiate non-central F-distribution object
    nq = rv.ppf(np.linspace(0,1,res)) # Get inverse bounds of standard distribution
    ncf_ps = 16*(psz-1)/nq # Invert for correct bounds
    
    x = np.linspace(min(ncf_ps[ncf_ps<np.inf]),max(ncf_ps[ncf_ps<np.inf]),res)
    f = interp1d(np.flip(ncf_ps),np.linspace(0,1,res))
    Y = f(x) 
    y=np.diff(Y)/(x[2]-x[1])
    return x[1:] , y

def find_invncf(d,s,psz,bnds=[0.025,0.975]):
    rv=st.ncf(1,psz-1,psz*d**2/(s**2))
    np=rv.ppf([1-bnds[1],1-bnds[0]])
    ncf_ps=16*(psz-1)/np
    return ncf_ps

def find_pilot_size(s,d,sprd,psz_init=5,pszmax=100010):
    found=False
    true_val=16*s**2/d**2
    while not found and psz_init<pszmax:
        ncf_ps=find_invncf(d,s,psz_init)
        if max(ncf_ps)<=(true_val+sprd) and min(ncf_ps)>=(true_val-sprd):
            found=True
        else:
            psz_init+=1
    return psz_init

def fii_datagen(res=1000,d=2,pszmax=110000):
    effect_size_vec=10**np.linspace(np.log10(5),-1,res)
    fii_data=np.zeros((res,3))
    
    i_N=np.ceil(16*effect_size_vec[0]**(-2))
    fii_data[0,0]=find_pilot_size(d/effect_size_vec[0],d,0.5)
    fii_data[0,1]=find_pilot_size(d/effect_size_vec[0],d,max(0.05*i_N,0.5))
    fii_data[0,2]=find_pilot_size(d/effect_size_vec[0],d,max(0.1*i_N,0.5))
    for ward in range(1,res):
        i_N=np.ceil(16*effect_size_vec[ward]**(-2))
        fii_data[ward,0]=find_pilot_size(d/effect_size_vec[ward],d,0.5,psz_init=fii_data[ward-1,0],pszmax=pszmax)
        fii_data[ward,1]=find_pilot_size(d/effect_size_vec[ward],d,max(0.5,0.05*i_N),psz_init=fii_data[ward-1,1],pszmax=pszmax)
        fii_data[ward,2]=find_pilot_size(d/effect_size_vec[ward],d,max(0.5,0.125*i_N),psz_init=fii_data[ward-1,2],pszmax=pszmax)
        
    return fii_data

def calculate_power(n, mu, sigma, alpha):
    df = 2 * (n - 1)
    delta = (mu) / (sigma / np.sqrt(n))
    critical_value = st.t.ppf(1 - alpha / 2, df)
    power = (1 - st.nct.cdf(critical_value, df, delta)) + st.nct.cdf(-critical_value, df, delta)
    return power
    
def plot_fig_1(fii_data,d=2,res=1000):
     
    fig, axs = plt.subplots(3, 4)
    fig.set_size_inches(16, 12)
    fig.subplots_adjust(wspace=0.2)
    
    # First panel (population parameters)
    cont_mean=0
    effect_size=1.5
    pop_var=0.5
    x_grid=np.linspace(-3,3.5+effect_size,res)
    y1=1/(np.sqrt(2*np.pi*pop_var))*np.exp(-(x_grid-cont_mean)**2/(2*pop_var))
    y2=1/(np.sqrt(2*np.pi*pop_var))*np.exp(-(x_grid-effect_size)**2/(2*pop_var))
    
    
    # axs[0,0].plot(cont_mean*np.ones(100),np.linspace(0,1/(np.sqrt(2*np.pi*pop_var)),100),'tab:grey',label='_nolegend_')
    # axs[0,0].plot(effect_size*np.ones(100),np.linspace(0,1/(np.sqrt(2*np.pi*pop_var)),100),'tab:grey',label='_nolegend_')
    axs[0,0].plot(x_grid,y1,colours_C[0],label='Cont.')
    axs[0,0].plot(x_grid,y2,colours_C[5],label='Treat.')
    axs[0,0].plot(np.linspace(cont_mean,effect_size,100),1/(np.sqrt(2*np.pi*pop_var))*np.ones(100),'black',label='_nolegend_')
    axs[0,0].plot(cont_mean*np.ones(100),1/(np.sqrt(2*np.pi*pop_var))*np.linspace(0.975,1.025,100),'black',label='_nolegend_')
    axs[0,0].plot(effect_size*np.ones(100),1/(np.sqrt(2*np.pi*pop_var))*np.linspace(0.975,1.025,100),'black',label='_nolegend_')
    axs[0,0].plot(np.linspace(cont_mean-np.sqrt(pop_var),cont_mean,100),1/(np.sqrt(2*np.pi*pop_var))*np.exp(-(np.sqrt(pop_var)-cont_mean)**2/(2*pop_var))*np.ones(100),'black',label='_nolegend_')
    axs[0,0].plot((cont_mean-np.sqrt(pop_var))*np.ones(100),1/(np.sqrt(2*np.pi*pop_var))*np.exp(-(np.sqrt(pop_var)-cont_mean)**2/(2*pop_var))*np.linspace(0.975,1.025,100),'black',label='_nolegend_')
    axs[0,0].plot((cont_mean)*np.ones(100),1/(np.sqrt(2*np.pi*pop_var))*np.exp(-(np.sqrt(pop_var)-cont_mean)**2/(2*pop_var))*np.linspace(0.975,1.025,100),'black',label='_nolegend_')
    
    axs[0,0].spines['top'].set_visible(False)
    axs[0,0].spines['right'].set_visible(False)
    axs[0,0].legend(['Cont.','Treat.'],frameon=False,loc='upper right',title=[],fontsize=11,title_fontsize=12)
    axs[0,0].set_xticks([])
    axs[0,0].set_yticks([])
    
    box = axs[0,0].get_position()
    box.x0 = box.x0 - 0.035
    box.x1 = box.x1 - 0.035
    box.y0 = box.y0 + 0.025
    box.y1 = box.y1 + 0.025
    axs[0,0].set_position(box)
    
    fig.text(0.17, 0.905, '$\mu$', ha='center', va='center',fontsize=12)
    fig.text(0.149, 0.8, '$\sigma$', ha='center', va='center',fontsize=12)
    fig.text(0.18, 0.65, 'Meas. val.', ha='center', va='center',fontsize=14)
    fig.text(0.065, 0.805, 'Density', va='center', rotation='vertical',fontsize=14)
    fig.text(0.045, 0.9, 'A',fontsize=20,weight='bold')
    
    
    
    # Second panel (Power and significance)
    mu0 = 0  # Mean of the null hypothesis
    mu1 = 2  # Mean of the alternative hypothesis
    sigma = 1  # Standard deviation
    alpha = 0.05  # Significance level
    x = np.linspace(-3, 6, 1000)
    null_dist = st.norm(mu0, sigma)
    alt_dist = st.norm(mu1, sigma)
    critical_value = null_dist.ppf(1 - alpha)
    axs[0,1].plot(x, null_dist.pdf(x), label='$H_0$', color=colours_A[0])
    axs[0,1].plot(x, alt_dist.pdf(x), label='$H_1$', color=colours_A[4])
    axs[0,1].fill_between(x, 0, null_dist.pdf(x), where=(x > critical_value), color=colours_A[0], alpha=1, label='α')
    axs[0,1].fill_between(x, 0, alt_dist.pdf(x), where=(x > critical_value), color=colours_A[4], alpha=0.2, label='1-β')
    axs[0,1].axvline(x=critical_value, color='black', linestyle='--', label='_nolegend_')
    axs[0,1].spines['top'].set_visible(False)
    axs[0,1].spines['right'].set_visible(False)
    axs[0,1].legend(['$ \mathrm{H}_0$','$ \mathrm{H}_1$','  α','1-β'],frameon=False,loc='upper right',title=[],fontsize=10,title_fontsize=12)
    axs[0,1].set_xticks([])
    axs[0,1].set_yticks([]) 
    fig.text(0.41, 0.65, 'Test statistic', ha='center', va='center',fontsize=14)
    fig.text(0.3, 0.805, 'Density', va='center', rotation='vertical',fontsize=14)
    fig.text(0.28, 0.9, 'B',fontsize=20,weight='bold')
    
    box = axs[0,1].get_position()
    box.y0 = box.y0 + 0.025
    box.y1 = box.y1 + 0.025
    axs[0,1].set_position(box)
    
    # Third panel (power 1)
    sample_sizes = np.arange(2, 16)
    powers_01 = [calculate_power(n, 2, 1, 0.1) for n in sample_sizes]
    powers_005 = [calculate_power(n, 2, 1, 0.05) for n in sample_sizes]
    powers_001 = [calculate_power(n, 2, 1, 0.01) for n in sample_sizes]
    powers_0001 = [calculate_power(n, 2, 1, 0.001) for n in sample_sizes]
    
    # Plotting the power curve
    axs[0,2].axhline(0.8, color='grey', linestyle='--', label='_nolegend_')
    axs[0,2].plot(sample_sizes, powers_01, label='α=0.1', color=colours_C[1])
    axs[0,2].plot(sample_sizes, powers_005, label='α=0.05', color=colours_C[2])
    axs[0,2].plot(sample_sizes, powers_001, label='α=0.01', color=colours_C[3])
    axs[0,2].plot(sample_sizes, powers_0001, label='α=0.001', color=colours_C[4])
    axs[0,2].spines['top'].set_visible(False)
    axs[0,2].spines['right'].set_visible(False)
    axs[0,2].legend(['  0.1',' 0.05',' 0.01','0.001'],frameon=False,loc=[0.6, 0.2],title='Sig. level α',fontsize=11,title_fontsize=12)
    axs[0,2].set_xticks([5,10,15])
    axs[0,2].set_xticklabels([5,10,15],fontsize=14)
    axs[0,2].set_yticks([0,0.4,0.8])
    axs[0,2].set_yticklabels([0,0.4,0.8],fontsize=14)
    axs[0,2].text(0.8, 0.15, '$\Delta$=2', transform=axs[0,2].transAxes, fontsize=14,
        verticalalignment='top')
    
    box = axs[0,2].get_position()
    box.x0 = box.x0 + 0.035
    box.x1 = box.x1 + 0.035
    box.y0 = box.y0 + 0.025
    box.y1 = box.y1 + 0.025
    axs[0,2].set_position(box)
    
    # Third panel (power 2)
    sample_sizes = np.arange(2, 51)
    powers_01 = [calculate_power(n, 1, np.sqrt(3), 0.1) for n in sample_sizes]
    powers_005 = [calculate_power(n,1, np.sqrt(3), 0.05) for n in sample_sizes]
    powers_001 = [calculate_power(n,1, np.sqrt(3), 0.01) for n in sample_sizes]
    powers_0001 = [calculate_power(n,1, np.sqrt(3), 0.001) for n in sample_sizes]
    

    # Plotting the power curve
    axs[0,3].axhline(0.8, color='grey', linestyle='--', label='_nolegend_')
    axs[0,3].plot(sample_sizes, powers_01, label='_nolegend_', color=colours_C[1])
    axs[0,3].plot(sample_sizes, powers_005, label='_nolegend_', color=colours_C[2])
    axs[0,3].plot(sample_sizes, powers_001, label='_nolegend_', color=colours_C[3])
    axs[0,3].plot(sample_sizes, powers_0001, label='_nolegend_', color=colours_C[4])
    axs[0,3].spines['top'].set_visible(False)
    axs[0,3].spines['right'].set_visible(False)
    axs[0,3].set_xticks([0,25,50])
    axs[0,3].set_xticklabels([0,25,50],fontsize=14)
    axs[0,3].set_yticks([0,0.4,0.8])
    axs[0,3].set_yticklabels([],fontsize=14)
    axs[0,3].text(0.75, 0.15, '$\Delta$=$\sqrt{2/3}$', transform=axs[0,3].transAxes, fontsize=14,
        verticalalignment='top')
    box = axs[0,3].get_position()
    box.x0 = box.x0 + 0.025
    box.x1 = box.x1 + 0.025
    box.y0 = box.y0 + 0.025
    box.y1 = box.y1 + 0.025
    axs[0,3].set_position(box)
    fig.text(0.75, 0.645, 'Sample size (N)', ha='center', va='center',fontsize=14)
    fig.text(0.52, 0.805, 'Power (1-β)', va='center', rotation='vertical',fontsize=14)
    fig.text(0.505, 0.9, 'C',fontsize=20,weight='bold')
    
    
    # Panel D (estimator distributions)
    s_vec=[1,np.sqrt(6)]
    psz_vec=[5,10,25,50,100]
    x_maxes= [10,60]
   
    y_maxes=np.zeros(2)
    for s_ind in range(2):
        s=s_vec[s_ind]
        n=np.ceil(round(16*s**2/d**2, 8))
        
        index=0
        y_max=0
        for psz in psz_vec:
            if s_ind==1:
                [x,y]=explicit_pdf(d,s,int(psz),res=1000,x_max=x_maxes[s_ind])
            else:
                [x,y]=stable_pdf(d,s,int(psz),res=1000,x_max=x_maxes[s_ind])
            #axs[s_ind,0].plot(x,y,colours_A[index])
            if max(y)>=y_max:
                y_max=max(y)
                y_maxes[s_ind]=y_max
            index+=1
        axs[s_ind+1,0].plot(n*np.ones(100),np.linspace(0,y_maxes[s_ind],100),color='black',linestyle=':',linewidth=1,label='_nolegend_')
        index=0
        for psz in psz_vec:
            if s_ind==1:
                [x,y]=explicit_pdf(d,s,int(psz),res=1000,x_max=x_maxes[s_ind])
            else:
                [x,y]=stable_pdf(d,s,int(psz),res=1000,x_max=x_maxes[s_ind])
            axs[s_ind+1,0].plot(x,y,colours_A[index])
            index+=1
        axs[s_ind+1,0].scatter(n,y_maxes[s_ind],50,'black',marker='*',label='_nolegend_')
        
        axs[s_ind+1,0].set_xlim(0,x_maxes[s_ind])
        [ymin,ymax]=axs[s_ind+1,0].get_ylim()
        axs[s_ind+1,0].set_ylim(0,ymax)
        
        axs[s_ind+1,0].spines['top'].set_visible(False)
        axs[s_ind+1,0].spines['right'].set_visible(False)

    axs[1,0].legend(['5','10','25','50','100'],bbox_to_anchor=[0.85,0.35],loc='center',frameon=False,title='Pilot size $n$',fontsize=11,title_fontsize=12)
    axs[1,0].set_xticks([0,4,8])
    axs[1,0].set_xticklabels([0,4,8],fontsize=14)
    axs[1,0].set_yticks([0,0.2,0.4,0.6])
    axs[1,0].set_yticklabels([0,0.2,0.4,0.6],fontsize=14)
    axs[1,0].text(0.7, 0.85, '$\Delta$=2', transform=axs[1,0].transAxes, fontsize=14,
        verticalalignment='top')
    axs[1,0].text(0.7, 0.95, '$N$=4', transform=axs[1,0].transAxes, fontsize=14,
        verticalalignment='top')
    box = axs[1,0].get_position()
    box.y0 = box.y0 - 0.025
    box.y1 = box.y1 - 0.025
    axs[1,0].set_position(box)
    
    axs[2,0].set_xticks([0,20,40,60])
    axs[2,0].set_xticklabels([0,20,40,60],fontsize=14)
    axs[2,0].set_yticks([0,0.02,0.04,0.06])
    axs[2,0].set_yticklabels([0,0.02,0.04,0.06],fontsize=14)
    #axs[1,0].set_xlabel('Estimated sample size',fontsize=14)
    axs[2,0].text(0.675, 0.85, '$\Delta$=$\sqrt{2/3}$', transform=axs[2,0].transAxes, fontsize=14,
        verticalalignment='top')
    axs[2,0].text(0.675, 0.95, '$N$=24', transform=axs[2,0].transAxes, fontsize=14,
        verticalalignment='top')
    
    
    fig.text(0.19, 0.0615, 'Estimated sample size ($\~N$)', ha='center', va='center',fontsize=14)
    fig.text(0.05, 0.35, 'Density', va='center', rotation='vertical',fontsize=14)
    fig.text(0.045, 0.605, 'D',fontsize=20,weight='bold')

    for s_ind in range(2):
        box = axs[s_ind+1,0].get_position()
        box.x0 = box.x0 - 0.03
        box.x1 = box.x1 - 0.03
        axs[s_ind+1,0].set_position(box)
    box = axs[2,0].get_position()
    box.y0 = box.y0 - 0.03
    box.y1 = box.y1 - 0.03
    axs[2,0].set_position(box)
    
    # Panel E (predictive intervals)
    s_vec=[1,np.sqrt(2),2,np.sqrt(6)]
    text_t=['$N$=4','$N$=8','$N$=16','$N$=24']
    text_b=['$\Delta$=2','$\Delta$=$\sqrt{2}$','$\Delta$=1','$\Delta$=$\sqrt{2/3}$']
    y_tick_list=[[1,2,5,10,20],[1,10,50,100],[1,10,100,1000],[1,10,100,1000]]
    y_tick_label_list=[['1','2','5','10','20'],['1','10','50','100'],['1','10','100','1000'],['1','10','100','1000']]
    
    plot_tuples=[(1,1),(1,2),(2,1),(2,2)]
    
    ax_ind=0
    for s in s_vec:
        n=np.ceil(round(16*s**2/d**2, 8))
        pilot_sizes=np.ceil(10**np.linspace(np.log10(3),3,100))
        for ward in range(1,100):
            if pilot_sizes[ward]<=pilot_sizes[ward-1]:
                pilot_sizes[ward]=pilot_sizes[ward-1]+1
        ncf_ps=np.zeros((100,4))
    
        index=0
        for psz in pilot_sizes:
            Z=find_invncf(d,s,psz)
            ncf_ps[index,0]=Z[0]
            ncf_ps[index,1]=Z[1]
            Z_quart=find_invncf(d,s,psz,bnds=[0.25,0.75])
            ncf_ps[index,2]=Z_quart[0]
            ncf_ps[index,3]=Z_quart[1]
            index+=1

        axs[plot_tuples[ax_ind]].fill_between(pilot_sizes,ncf_ps[:,3],ncf_ps[:,1],color=colours_A[0],alpha=0.25,label=' 95% pred.')
        axs[plot_tuples[ax_ind]].fill_between(pilot_sizes,ncf_ps[:,2],ncf_ps[:,3],color=colours_A[0],alpha=0.95,label=' 50% pred.')
        axs[plot_tuples[ax_ind]].fill_between(pilot_sizes,ncf_ps[:,0],ncf_ps[:,2],color=colours_A[0],alpha=0.25,label='_nolegend_')
        
        axs[plot_tuples[ax_ind]].loglog(
            pilot_sizes,n*np.ones(100),colours_A[4],label='True size $N$')
        axs[plot_tuples[ax_ind]].loglog(
            pilot_sizes,ncf_ps[:,0],colours_A[0],linewidth=1,label='_nolegend_')
        axs[plot_tuples[ax_ind]].loglog(
            pilot_sizes,ncf_ps[:,1],colours_A[0],linewidth=1,label='_nolegend_')
        
        
        axs[plot_tuples[ax_ind]].spines['top'].set_visible(False)
        axs[plot_tuples[ax_ind]].spines['right'].set_visible(False)
        [ymin,ymax]=axs[plot_tuples[ax_ind]].get_ylim()
        
        axs[plot_tuples[ax_ind]].set_ylim([1,ymax])
        axs[plot_tuples[ax_ind]].tick_params(axis='both',labelsize=14)
        if ax_ind<=1:
            axs[plot_tuples[ax_ind]].xaxis.set_ticklabels([])
        axs[plot_tuples[ax_ind]].set_yticks(y_tick_list[ax_ind])
        axs[plot_tuples[ax_ind]].set_yticklabels(y_tick_label_list[ax_ind],fontsize=14)
        axs[plot_tuples[ax_ind]].text(0.7, 0.95, text_t[ax_ind], transform=axs[plot_tuples[ax_ind]].transAxes, fontsize=14,
            verticalalignment='top')
        axs[plot_tuples[ax_ind]].text(0.7, 0.85, text_b[ax_ind], transform=axs[plot_tuples[ax_ind]].transAxes, fontsize=14,
            verticalalignment='top')
        if plot_tuples[ax_ind][0]==2:
            box = axs[plot_tuples[ax_ind]].get_position()
            box.y0 = box.y0 - 0.03
            box.y1 = box.y1 - 0.03
            axs[plot_tuples[ax_ind]].set_position(box)
        else:
            box = axs[plot_tuples[ax_ind]].get_position()
            box.y0 = box.y0 - 0.025
            box.y1 = box.y1 - 0.025
            axs[plot_tuples[ax_ind]].set_position(box)
        if plot_tuples[ax_ind][1]==1:
            box = axs[plot_tuples[ax_ind]].get_position()
            box.x0 = box.x0 + 0.01
            box.x1 = box.x1 + 0.01
            axs[plot_tuples[ax_ind]].set_position(box)
        ax_ind+=1
    axs[plot_tuples[1]].legend(frameon=False,fontsize=10,loc='lower right')
    
    fig.text(0.282, 0.35, 'Estimated sample size ($\~N$)', va='center', rotation='vertical',fontsize=14)
    fig.text(0.5, 0.06, 'Pilot study size ($n$)', ha='center', va='center',fontsize=14)
    fig.text(0.27, 0.605, 'E',fontsize=20,weight='bold')
    
   # Panel F (Predictive interval width and necessary pilot size)
    res=1000
    pilot_sizes=[10,100,1000]
    effect_size_vec=10**np.linspace(np.log10(5),-1,res)
    labels=['Exact','10%','25%']
    
    pred_at_size=np.zeros((3,res))
    for sooth in range(3):
        for ward in range(res):
            ncf_ps=find_invncf(d,d/effect_size_vec[ward],pilot_sizes[sooth])
            pred_at_size[sooth,ward]=ncf_ps[0]-ncf_ps[1]
    
    for sooth in range(3):
        axs[1,3].loglog(effect_size_vec,pred_at_size[sooth,:],colours_C[2*sooth])
    
    
    for sooth in range(3):
       axs[2,3].loglog(effect_size_vec,fii_data[:,2-sooth],colours_C[2*(2-sooth)+1],label=labels[sooth])
   
    axs[1,3].spines['top'].set_visible(False)
    axs[1,3].spines['right'].set_visible(False)
    axs[1,3].legend(['  10',' 100','1000'],frameon=False,loc='lower right',title='Pilot size $n$',fontsize=11,title_fontsize=12)
    axs[1,3].set_xticks([1,0.1])
    axs[1,3].set_xticklabels(['$10^0$','$10^{-1}$'],fontsize=14)
    axs[1,3].set_xlim([0.1,5])
    axs[1,3].set_yticks([1,10,100,1000,10000])
    axs[1,3].set_yticklabels(['$10^0$','$10^1$','$10^2$','$10^3$','$10^4$'],fontsize=14)
    axs[1,3].set_ylim([1,10000])
    axs[1,3].set_ylabel('Pred. interval width',fontsize=14)
    axs[1,3].invert_xaxis()

    axs[2,3].spines['top'].set_visible(False)
    axs[2,3].spines['right'].set_visible(False)
    axs[2,3].set_xticks([1,0.1])
    axs[2,3].set_xticklabels(['$10^0$','$10^{-1}$'],fontsize=14)
    axs[2,3].set_xlim([0.1,5])
    axs[2,3].set_yticks([10,100,1000,10000,100000])
    axs[2,3].set_yticklabels(['$10^1$','$10^2$','$10^3$','$10^4$','$10^5$'],fontsize=14)
    axs[2,3].set_ylim([10,100000])
    handles, labels = axs[2,3].get_legend_handles_labels()
    axs[2,3].legend(handles[::-1],labels,loc='lower right',frameon=False,title='Pred. interval',fontsize=11,title_fontsize=12)
    axs[2,3].set_ylabel('Pilot study size ($n$)',fontsize=14)

    axs[2,3].invert_xaxis()
   
    fig.text(0.845, 0.06, 'Stand. effect size ($\Delta$)', ha='center', va='center',fontsize=14)
    fig.text(0.695, 0.605, 'F',fontsize=20,weight='bold')
    
    for s_ind in range(2):
        box = axs[s_ind+1,3].get_position()
        box.x0 = box.x0 + 0.025
        box.x1 = box.x1 + 0.025
        axs[s_ind+1,3].set_position(box)
    box = axs[1,3].get_position()
    box.y0 = box.y0 - 0.025
    box.y1 = box.y1 - 0.025
    axs[1,3].set_position(box)
    box = axs[2,3].get_position()
    box.y0 = box.y0 - 0.03
    box.y1 = box.y1 - 0.03
    axs[2,3].set_position(box)
    # Save figure
    fig.set_dpi(900)
    fig.savefig('fig_1.pdf',bbox_inches='tight')
    fig.savefig('fig_1.png',bbox_inches='tight')
    fig.savefig('fig_1.tif',bbox_inches='tight')
    
try:
    plot_fig_1(fii_data)
except:
    fii_data=fii_datagen()
    plot_fig_1(fii_data)
    
        