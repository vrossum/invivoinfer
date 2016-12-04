from __future__ import division
import numpy as np
np.seterr(divide = 'warn')

import scipy as sp
import scipy.stats as pyst
import matplotlib.pyplot as plt

import invivoinfer.mc_functions as mcf
import invivoinfer.pyprocess as SP

class VCGeneration:

    #---------------- CONSTRUCTOR
    def __init__(self, Syn_exc_par,Syn_inh_par,Noise_par,Fr_par,Init_par,Sim_par):
         '''
         This class generates an object that represent a Vclamp simulation. 
         In order to be constructed it needs the following parameters
         '''
         self.Syn_exc_par = Syn_exc_par
         self.Syn_inh_par = Syn_inh_par
         self.Noise_par = Noise_par
         self.Fr_par = Fr_par
         self.Init_par = Init_par
         self.Sim_par = Sim_par
         
         self.Syn_exc_par['tau1_std'] = self.Syn_exc_par.get('tau1_std', 0.)
         self.Syn_exc_par['tau2_std'] = self.Syn_exc_par.get('tau2_std', 0.)


 #---------------- reset values
   
    def reset(self,tstop='def'): # default time is the one indicated in the main
        '''
        resets to default or zero: time, input times, input amplitudes, current.
        '''

        dt=self.Sim_par['dt'];
        if tstop=='def':
            tstop=self.Sim_par['duration'];
        self.timevec=np.array(np.arange(0,tstop,dt)) # reset the time

        self.input_time=[] #reset the input time
        self.input_amplitude=[] #.. and amplitudes
        self.I=np.zeros(self.timevec.shape) # and current

    def run(self,tstop='def',rate='def'):
        if tstop=='def':
            tstop=self.Sim_par['duration']
        if rate=='def':
            rate=self.Syn_exc_par['freq']
        
        self.reset(tstop)

        self.input_time=mcf.inhom_poisson(rate,self.timevec) # generate the timing of the poisson spikes
      
        if self.Syn_exc_par['WeightDist']=='LogNormal': # generate the amplitudes of each spike
            if self.Syn_exc_par['WeightType']=='Moments':
                B_mu_e=np.log((self.Syn_exc_par['A']**2)/np.sqrt(self.Syn_exc_par['stdA']**2+self.Syn_exc_par['A']**2))
                B_sigma_e=np.sqrt(np.log(self.Syn_exc_par['stdA']**2/self.Syn_exc_par['A']**2 +1))
            elif self.Syn_exc_par['WeightType']=='Parameters':
                B_mu_e=self.Syn_exc_par['A']
                B_sigma_e=self.Syn_exc_par['stdA']
            else:
                raise Exception('weighttype {} not permitted'.format(self.Syn_exc_par['WeightType']))

            if B_sigma_e > 0:
                self.input_amplitude=np.random.lognormal(B_mu_e,B_sigma_e,sp.size(self.input_time))
            else:
                self.input_amplitude=np.random.lognormal(B_mu_e,10**-10,sp.size(self.input_time))

        elif self.Syn_exc_par['WeightDist']=='TruncNormal': # generate the amplitudes of each spike
            if self.Syn_exc_par['WeightType']=='Moments':
                B_mu_e, B_sigma_e = mcf.mom2par(self.Syn_exc_par['A'],self.Syn_exc_par['stdA'],distType=self.Syn_exc_par['WeightDist'])

            elif self.Syn_exc_par['WeightType']=='Parameters':
                B_mu_e=self.Syn_exc_par['A']
                B_sigma_e=self.Syn_exc_par['stdA']
            else:
                raise Exception('weighttype {} not permitted'.format(self.Syn_exc_par['WeightType']))

            if B_sigma_e > 0:
                a, b = (0 - B_mu_e) / B_sigma_e, (10**10 - B_mu_e) / B_sigma_e

                self.input_amplitude=pyst.truncnorm.rvs(loc=B_mu_e,scale=B_sigma_e,a=a,b=b,size=sp.size(self.input_time))
            else:
                self.input_amplitude=self.Syn_exc_par['A'] * np.ones(sp.size(self.input_time))

        elif self.Syn_exc_par['WeightDist']=='Exponential': 
            if self.Syn_exc_par['WeightType']=='Moments':
                par1, par2 = mcf.mom2par(self.Syn_exc_par['A'],self.Syn_exc_par['stdA'],distType=self.Syn_exc_par['WeightDist'])

            elif self.Syn_exc_par['WeightType']=='Parameters':
                par1 = self.Syn_exc_par['A'];
                par2 = self.Syn_exc_par['stdA'];

            if self.Syn_exc_par['stdA'] > 0:
                
                ef = mcf.str_exp()
                self.input_amplitude = ef.rvs(size = sp.size(self.input_time), p1 = par1, p2 = par2)
                
            else:
                self.input_amplitude=self.Syn_exc_par['A'] * np.ones(sp.size(self.input_time))


        else:
            print 'ERROR, NOT AN ALLOWED DISTRIBUTION';

        if self.Syn_exc_par['tau1_std'] == 0 :
            tau1 = self.Syn_exc_par['tau1']
        else :
            tau_mu_e, tau_sigma_e = mcf.mom2par(self.Syn_exc_par['tau1'], self.Syn_exc_par['tau1_std'],
                                            distType='TruncNormal')
            a, b = (0 - tau_mu_e) / tau_sigma_e, 10**10

            tau1 = pyst.truncnorm.rvs(loc = tau_mu_e, scale = tau_sigma_e, a = a, b = b, size = sp.size(self.input_time))

        if self.Syn_exc_par['tau2_std'] == 0 :
            tau2 = self.Syn_exc_par['tau2']
        else :
            tau_mu_e, tau_sigma_e = mcf.mom2par(self.Syn_exc_par['tau2'], self.Syn_exc_par['tau2_std'],
                                                distType='TruncNormal')
            a, b = (0 - tau_mu_e) / tau_sigma_e, 10 ** 10

            tau2 = pyst.truncnorm.rvs(loc=tau_mu_e, scale=tau_sigma_e, a=a, b=b, size=sp.size(self.input_time))

        
        self.I=eventgen(self.input_time,tau1,tau2,self.input_amplitude,self.I,self.Sim_par['dt'])
        
        if self.Noise_par['ampli'] != 0:
            self.add_OU_noise(self.Noise_par['mu'],self.Noise_par['ampli'],self.Noise_par['nu_cutoff'])


    def add_OU_noise(self,mu,A,nu_cutoff):
        # add to a raw_trace a OU noise, of the form dOU_t = theta*(mu-OU_t)*dt + sigma*dB_t$;
        # parametrised as dOU_t = 2*pi/nu_cutoff*(mu-OU_t)*dt + A*sqrt(4*pi*nu_coff)*dB_t$, to have a process of std=A and powerspectrum cut at v_coff
        tstop=self.Sim_par['duration'];
        time_vec=np.array(sp.arange(0,tstop,self.Sim_par['dt']))
        start_pos=mu
        theta=2*np.pi*nu_cutoff
        sigma=A*np.sqrt(4*np.pi*nu_cutoff)

        OU = SP.OU_process(theta,mu,sigma,0,start_pos)
        ou_trace=OU.sample_path(time_vec).T
        ou_trace=ou_trace[:,0]
        
        self.I=self.I+ou_trace


#---------------- plots

    def plot_I(self):
        plt.subplot(1,2,1)
        plt.plot(self.timevec,self.I)
        plt.xlabel('Time')
        plt.ylabel('Current (pA)')

        plt.subplot(1,2,2)
        num_bins=100
        n, bins, patches = plt.hist(self.I, num_bins, normed=1, facecolor='green', alpha=0.5)
        plt.xlabel('Current (pA)')
        plt.ylabel('Prob.')
        plt.show()

# -------------- FUNCTION USED BY THE CLASS

def meanstd2lognorm(a,b):
    aln=log((a**2)/(sqrt((b)**2 + a**2)))
    bln=sqrt(log((b)**2/(a**2)+1))

    return aln, bln


def eventgen(input_time,t1,t2,A,itrace,dt,TrueAmplitude=False):
    ''' 
    TrueAmplitude=True, means that the events are normalised so to have the peak equal to amplitude.
    TrueAmplitude=False, means that the peak will be lower than the amplitude, unless t1<<t2
    '''
    if TrueAmplitude==True:
        A=A/((t2/(t2+t1))*(t1/(t1+t2))**(t1/t2))
    
    input_bin=np.around(input_time/dt).astype(int)
    
    input_conv=np.zeros(sp.size(itrace))
    
    
    if sp.size(t1) == 1 :
        input_conv[input_bin]=A
    
        x=np.arange(0, t2*5, dt)
        input_kernel = (1 - np.exp(-x/t1))*np.exp(-x/t2)

        out = np.convolve(input_conv,input_kernel,'same')
    else:
        
        x=np.arange(0, t2.mean()*5, dt)
        input_conv = np.hstack([input_conv, np.zeros(sp.size(x))])
        for i, val in enumerate(A):
            inp = val*(1 - np.exp(-x/t1[i]))*np.exp(-x/t2[i])

            input_conv[input_bin[i] : input_bin[i] + sp.size(inp)] = input_conv[input_bin[i] : input_bin[i] + sp.size(inp)] + inp
    
        out = input_conv[:sp.size(itrace)]

    
    return out
