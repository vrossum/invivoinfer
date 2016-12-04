from __future__ import division

import numpy as np
np.seterr(divide = 'warn')
import scipy.io as spio
import scipy.stats as pyst
import pandas as pd

import seaborn as sns

import sys,os

import vc_gen_class as vgc
import vc_infer_class_rewrite as vic
import plot_scripts as plt_s

import mc_functions as mcf
import signalproc_functions as spf

import pdb

import pandas as pd

import time
import matplotlib.pyplot as plt

import shelve
import pickle

def vc_infer_rawdata(c_id,c_type='interneurons',parameters=[]):

    # Path to database and db name
    path='/Users/paolopuggioni/invivoinfer/raw_data/db/'
    filenamedb='db_vc.h5'
    db=pd.HDFStore(path+filenamedb)


    # Load file
    rawdata=spio.loadmat('/Users/paolopuggioni/invivoinfer/raw_data/{}_in_raw.mat'.format(c_id))
    start_time=db['interneurons']['StartT'][c_id]
    stop_time=db['interneurons']['StopT'][c_id]

    rawdata=spio.loadmat('/Users/paolopuggioni/invivoinfer/raw_data/{}_in_raw.mat'.format(c_id))

    # get start and end of the relevant raw_trace
    start_time=db['interneurons']['StartT'][c_id]
    stop_time=db['interneurons']['StopT'][c_id]

    # get data in the correct shape
    for nh in iter(rawdata):
        try:
            if np.size(rawdata[nh],axis=1)>1:
                rawdata[nh]=rawdata[nh].T
        except: pass
        try:
            rawdata[nh]=rawdata[nh][:,0]
        except: pass

    # cut the raw_trace to relevant time

    trace=rawdata['vc'][(rawdata['time']>start_time) & (rawdata['time']<stop_time)]
    speed=rawdata['speed'][(rawdata['time']>start_time) & (rawdata['time']<stop_time)]
    dt=rawdata['time'][1]-rawdata['time'][0]
    time=np.linspace(0,dt*np.size(trace,axis=0)-dt,np.size(trace,axis=0))

    motion=rawdata['motion'][(rawdata['videotime']>start_time) & (rawdata['videotime']<stop_time)]
    dt_m=rawdata['videotime'][1]-rawdata['videotime'][0]
    time_motion=np.linspace(0,dt_m*np.size(motion,axis=0)-dt_m,np.size(motion,axis=0))

    # GET MOVEMENT AND QUIET TRACES
    movthr=db['interneurons']['movthr'][c_id]
    mov_break=0.5 # in seconds if motion below thr for this long, still considered movement
    
    trace_quiet, trace_mov, mov_start_time, mov_stop_time = mov_split(trace,dt,motion,dt_m,movthr,
                                                                      mov_break=mov_break, verbose=0)

    
    # plot looking for spikes
    # spike_cut(trace, dt, 1, verbose=parameters['verbose'])

    # defining types of loops
    trace_infer=['quiet','movement']
    infer_distType=['LogNormal','TruncNormal','Exponential']
    #infer_distType=['LogNormal']
    # plot 
    plot_vc_distr(trace_quiet,trace_mov,dt)


    # GLOBAL TAU ESTIMATION
    gl_tau = infer_tau(trace, dt, taus_estimation_par=parameters['taus_estimation_par'], verbose = parameters['verbose'])


    # ALL PARAMETER INFERENCE

    data={}
    for inf_dist in iter(infer_distType):
        data[inf_dist]={'quiet':{},'movement':{}}

    for b_state in iter(trace_infer):
        print '-------------------  ANALYSING THE {} PART\n\n'.format(b_state)
        
        for inf_dist in iter(infer_distType):
            print '-------------------  USING THE {} WEIGHT DISTRIBUTION\n\n'.format(inf_dist)
            vc=[]
            if b_state=='quiet':
                tr=-trace_quiet
            elif b_state=='movement':
                tr=-trace_mov
            # START THE INFERENCE                
            vc=vic.VCInfer(tr,dt)
            #BASIC QUANTITIES
            vc.momenta(verbose=parameters['verbose'])
            #calculate the timewindow
            t_time=dt*np.size(tr)
            wl=t_time/20.

            if wl>2.:
                wl=2.
            elif wl<1:
                wl=1.
            print '\nPSD calculated with {} time window\n'.format(wl)

            vc.psd_calc(windowl=wl,verbose=parameters['verbose'])
            # ESTIMATING TAU
            vc.taus_init(parameters['taus_estimation_par'])
            vc.estimate_taus(verbose=parameters['verbose'],plot=0)
        
            # RESET THE UNCERTAINTIES
            vc.reset_obs_unc()
            # THIS IS FOR THE MEAN
            base_line_force={'average':-db['interneurons']['I_base'][c_id],
                             'uncertainty':db['interneurons']['I_unc'][c_id], 
                             'ToUse':True} # IF TRUE you force the baseline to the set value
            vc.set_obs_unc_mean(base_line_force,verbose=parameters['verbose'])

            # THIS IS FOR THE STANDARD DEVIATION
            vc.set_obs_unc_std(parameters['hf_std_corr'],parameters['lf_std_corr'],verbose=parameters['verbose'],plot=0) # VERBOSE 1 print, 2 plot        
            #HERE I USE THE GLOBAL TAU
            if parameters['taus_estimation_par']['tau_touse']=='G':
                vc.param_priors=gl_tau

            # -------------- FIRST GUESS
            # first guess param
            
            parameters['first_guess_par']['distType']=inf_dist

   
            if inf_dist == 'Exponential': # This is a hack to avoid useless high values of the index.
                parameters['first_guess_par']['stdA_bound'] = [0.5, 15.]

            vc.first_guess_init(first_guess_par=parameters['first_guess_par'])
            vc.first_guess(verbose=parameters['verbose'])
        
            # ----- OPTIMIZE LIKELIHOOD           
            vc.optimize_likelihood(kernel_par=parameters['kernel_par'],likelihood_par=parameters['likelihood_par'],
                                   sk_ku_corr=parameters['sk_ku_corr'],verbose=parameters['verbose'])


            # ----- Re-initialise the MC
            parameters['prior_bound']['tau2_bound']=['Normal',vc.param_priors['tau2'][0],vc.param_priors['tau2'][1],0,0.01]
            parameters['prior_bound']['tau1_bound']=['Normal',vc.param_priors['tau1'][0],vc.param_priors['tau1'][1],0,0.005]

            if inf_dist == 'Exponential': # This is a hack to avoid useless high values of the index.
                parameters['prior_bound']['stdA_bound'] = ['Exponential', 10.]

            vc.model_init(prior_bound=parameters['prior_bound'],distType=inf_dist,epsilon=0.)

            # ----- Create the model
            vc.create_model(verbose=parameters['verbose'])

            # ------------------------ BAYESIAN INFERENCE
            vc.sampler_init(map_par=parameters['map_par'],mc_mh_par=parameters['mc_mh_par'])

            true_values=False
            vc.run_sampler(sampler_type=parameters['sampler_type'],true_values=False,verbose=parameters['verbose'])
            

            # save data
            data[inf_dist][b_state]['samples'] = vc.inference_results[parameters['sampler_type']]
            data[inf_dist][b_state]['maps'] = vc.inference_results['map']
            data[inf_dist][b_state]['bic'] = vc.inference_results['map']['ms']['bic']
            data[inf_dist][b_state]['aic'] = vc.inference_results['map']['ms']['aic']
            data[inf_dist][b_state]['dic'] = vc.inference_results[parameters['sampler_type']]['dic']


    # SAVE DATA
    path='../raw_data/analysis/'
    filename= '{}_analysis.pkl'.format(c_id)

    try:
        os.system('cp {}{} {}OLD_{}'.format(path,filename,path,filename))
    except:
        pass
    
    with open(path+filename, 'wb') as handle:
        pickle.dump(data, handle)

    print 'Saving file {}{}'.format(path,filename)

    # PRINT MAP
    summary_map={}
    index_val=['A','stdA','freq','tau1','tau2','aic','bic','dic']

    index_val_mc=['A','stdA','freq','tau1','tau2']
    summary_mc={}
    for inf_dist in iter(infer_distType):
        summary_mc[inf_dist]={s:{} for s in index_val_mc}
        for par in iter(summary_mc[inf_dist]):
            summary_mc[inf_dist][par]={'quiet':[],'movement':[]}

    for inf_dist in iter(infer_distType):
        summary_map[inf_dist]=pd.DataFrame(columns=['quiet','movement'],index=index_val)
        #pdb.set_trace()
        for b_state in iter(trace_infer):
            for ind in iter(index_val):
                #pdb.set_trace()
                try:
                    summary_map[inf_dist][b_state][ind]=data[inf_dist][b_state]['maps']['par'][ind].values[0]            
                except:
                    pass
                
                try:
                    summary_map[inf_dist][b_state][ind]=data[inf_dist][b_state]['maps']['useful'][ind].values[0]
                except:
                    pass
                
                try:
                    summary_map[inf_dist][b_state][ind]=data[inf_dist][b_state]['maps']['ms'][ind]
                except:
                    pass
         
                try:
                    summary_mc[inf_dist][ind][b_state]=data[inf_dist][b_state]['samples']['all'][ind]
                except:
                    pass
            
                try: 
                    summary_mc[inf_dist][ind][b_state]=data[inf_dist][b_state]['samples']['useful'][ind]
                except:
                    pass

                try: 
                    summary_mc[inf_dist][ind][b_state]=data[inf_dist][b_state]['samples'][ind]
                except:
                    pass
            
        print '\n\n{} weight distribution:'.format(inf_dist)                    
        print summary_map[inf_dist]


    # SAVE MAP

    filename= '{}_map.pkl'.format(c_id)

    try:
        os.system('cp {}{} {}OLD_{}'.format(path,filename,path,filename))
    except:
        pass
    
    with open(path+filename, 'wb') as handle:
        pickle.dump(summary_map, handle)

    print 'Saving file {}{}'.format(path,filename)
    
    # SAVE MC

    filename= '{}_mc.pkl'.format(c_id)

    try:
        os.system('cp {}{} {}OLD_{}'.format(path,filename,path,filename))
    except:
        pass
    
    with open(path+filename, 'wb') as handle:
        pickle.dump(summary_mc, handle)

    print 'Saving file {}{}'.format(path,filename)

    # PLOT SAVE DISTRIBUTIONS
    plot_mc_summary(c_id,summary_mc)
    
    a = 1
    return a 

def spike_cut(trace, dt, threshold, verbose=1):
    '''
    at the moment just does the plot, but does not actually cut the spikes
    '''
    
    if verbose>0:
        dv=np.diff(trace)/dt/1000
        dv=np.hstack([dv,dv[-1]])
        plt.plot(trace,dv)
        plt.xlabel(r'I_c (pA)')
        plt.ylabel(r'\Delta I_c/ \Delta t (pA/ms)')



def mov_split(trace,dt,motion,dt_m,movthr,mov_break=0.5, verbose=1):
    '''
    given 
    - a raw_trace with timestep dt,
    - a motion index with timestep dt_m
    - a movement threshold
    - a mov_break: in seconds if motion below thr for this long, still considered movement

    returns:
    - quiet raw_trace
    - movement raw_trace
    - start-stop of movement

    if verbose, plots some relevant data
    '''
    
    time=np.linspace(0,dt*np.size(trace,axis=0)-dt,np.size(trace,axis=0))
    time_motion=np.linspace(0,dt_m*np.size(motion,axis=0)-dt_m,np.size(motion,axis=0))
    
    mov_break_bins=np.ceil(mov_break/dt_m)
    mov_binary=np.zeros(np.size(motion))
    movindex=np.where(motion>movthr)
    movindex=movindex[0]
    mov_binary[movindex]=1.

    mov_start=np.where(np.diff(mov_binary)==1)[0]+1
    mov_stop=np.where(np.diff(mov_binary)==-1)[0]+1

    mov_start_time=time_motion[mov_start]
    mov_stop_time=time_motion[mov_stop]


    if mov_start_time[0]<mov_stop_time[0]: # the mouse is initially quiet
        if np.size(mov_start_time)==np.size(mov_stop_time): #the mouse ends being quiet
            ind=np.where(mov_start_time[1:]-mov_stop_time[:-1] <mov_break)[0]
            mov_stop_time=np.delete(mov_stop_time, ind)
            mov_start_time=np.delete(mov_start_time, ind+1)
        
        else: # the mouse ends moving
        
            ind=np.where(mov_start_time[1:]-mov_stop_time[:] <mov_break)[0]
            mov_stop_time=np.delete(mov_stop_time, ind)
            mov_start_time=np.delete(mov_start_time, ind+1)
        
            mov_stop_time=np.hstack([mov_stop_time,time_motion[-1]])


    else: # the mouse is initially moving
        if np.size(mov_start_time)==np.size(mov_stop_time): #the mouse ends being moving
            ind=np.where(mov_start_time[:]-mov_stop_time[:] <mov_break)[0]
            mov_stop_time=np.delete(mov_stop_time, ind)
            mov_start_time=np.delete(mov_start_time, ind)
            mov_stop_time=np.hstack([mov_stop_time,time_motion[-1]])
        
        else: # the mouse ends being quiet
            ind=np.where(mov_start_time[:]-mov_stop_time[:-1] <mov_break)[0]
            mov_stop_time=np.delete(mov_stop_time, ind)
            mov_start_time=np.delete(mov_start_time, ind)
        
        mov_start_time=np.hstack([dt_m,mov_start_time])


    index_mov=[0]
    if mov_start_time[0]>dt_m:
        index_quiet=np.where(time<mov_start_time[0])[0]
    else:
        index_quiet=[0]
    for i,val in enumerate(mov_start_time):
        temp=np.where((time>mov_start_time[i])&(time<mov_stop_time[i]))[0]
    #pdb.set_trace()
        index_mov=np.hstack([index_mov,temp])
    for i,val in enumerate(mov_start_time[:-1]):
        temp=np.where((time<mov_start_time[i+1])&(time>mov_stop_time[i]))[0]
        index_quiet=np.hstack([index_quiet,temp])

    m_ind=np.zeros(np.size(time))
    m_ind[index_mov]=1


    trace_mov=trace[m_ind==1]
    trace_quiet=trace[m_ind==0]

    if verbose>0:
        print '\n\nMOVEMENT / QUIET PLOTS\n'
        
        plt.subplot(211)
        plt.plot(time,trace)
        plt.subplot(212)
        plt.plot(time_motion,motion)
        plt.plot(time_motion,np.ones(np.size(time_motion))*movthr,'r')
        plt.plot(mov_start_time,np.zeros(np.size(mov_start_time)),'.r')
        plt.plot(mov_stop_time,np.zeros(np.size(mov_stop_time)),'.g')
        plt.show()

        plt.subplot(111)
        plt.plot(trace_mov)
        plt.plot(trace_quiet)
        plt.show()


    return trace_quiet, trace_mov, mov_start_time, mov_stop_time



def plot_vc_distr(trace_quiet,trace_mov,dt):
    trace_infer=['quiet','movement']
    color_p=['blue','red']

    for i, val in enumerate(trace_infer):
        if val=='quiet':
            tr=-trace_quiet
        elif val=='movement':
            tr=-trace_mov
        
        t_i=np.linspace(0,dt*np.size(tr,axis=0),np.size(tr))

        plt.figure()
        plt.subplot(np.size(trace_infer),2,2*i+1)    
        plt.plot(t_i,tr)
    
        plt.subplot(np.size(trace_infer),2,2*i+2)
        sns.kdeplot(tr,shade=True,vertical=True,color=color_p[i], label=val)


def infer_tau(trace, dt,
              taus_estimation_par = {'tau2_bound' : [0,0.01], 'tau1_bound' : [0,0.005],
                                   'lim_fit_std' : [10,500], 'lim_fit_psd' : [10,500],
                                   'n_iter_tau' : 30},
                verbose = 0.):
    '''
    estimates only tau and returns an .param_priors [it will be used for the other inference]
    '''
    #GET GLOBAL TAU
    #calculate the timewindow
    t_time=dt*np.size(trace)
    wl=t_time/20.

    if wl>2.:
        wl=2.
    elif wl<1.:
        wl=1.
    print '\nPSD calculated with {} time window\n'.format(wl)

    vt=vic.VCInfer(-trace,dt)
    vt.momenta(verbose=verbose)
    vt.psd_calc(windowl=wl,verbose=verbose)
    vt.taus_init(taus_estimation_par)
    vt.estimate_taus(verbose=verbose,plot=verbose)

    return vt.param_priors




def plot_mc_summary(c_id,summary_mc):



    import matplotlib.gridspec as gridspec

    sns.set(style="ticks",context='paper')

    color_p=['blue','red']
    infer_distType=['LogNormal','TruncNormal']
    index_val_mc=['A','stdA','freq','tau1','tau2']
    trace_infer=['quiet','movement']
    for inf_dist in iter(infer_distType): 
        plt.figure()
        for j, par in enumerate(index_val_mc):
            plt.subplot(np.size(index_val_mc),1,j)
            for i,b_state in enumerate(trace_infer):      
                sns.kdeplot(summary_mc[inf_dist][par][b_state], shade=True, color=color_p[i],label='')
                frame1 = plt.gca()        
                frame1.axes.get_yaxis().set_visible(False)
            if par == 'A':
                plt.xlim([0,100])
                plt.xticks(np.arange(0,110,20))
                plt.xlabel('{} (pA)'.format(par))
            elif par == 'stdA':
                plt.xlim([0,100])
                plt.xticks(np.arange(0,110,20))
                plt.xlabel('{} (pA)'.format(par))
            elif par == 'freq':
                plt.xlim([0,1500])
                plt.xticks(np.arange(0,1600,300))
                plt.xlabel('{} (Hz)'.format(par))
            elif par == 'tau1':
                plt.xlim([0,0.001])
                plt.xticks(np.arange(0,0.0011,0.0002),np.arange(0,1.1,0.2))
                plt.xlabel(r'$\tau_1$ (ms)')
            elif par == 'tau2':
                plt.xlim([0,0.004])
                plt.xticks(np.arange(0,0.0051,0.001),np.arange(0,5.1,1))
                plt.xlabel(r'$\tau_2$ (ms)')
        
        #plt.yticks([])
        sns.despine()
        fig_size=[1.7,4]
        fig_name=plt.gcf()
        fig_name.set_size_inches(fig_size)
        plt.tight_layout(pad=1.1)
        plt_s.save_plot('{}_{}_Posterior_param.pdf'.format(c_id,inf_dist),fig_size=fig_size,file_format='pdf')
    
