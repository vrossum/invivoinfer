from __future__ import division
import numpy as np
np.seterr(divide = 'warn')

import scipy.stats as pyst
import scipy.signal as sps
import scipy.optimize as sopt
import scipy.special as ss

import random

import matplotlib.pyplot as plt

# --------------------   Generate Inhomogeneous Poisson Process time events

def inhom_poisson(rate,t,rmax=10000.):
    # rate is the rate of the poisson process. If it's a number, the process is homogeneous, if it's a vector has to have the same size as t
    # t is a vector containing all the time bins
    # rmax is the maximum rate that the poisson process can have

    

    dt=t[2]-t[1]

    if np.size(rate)==1:
        rgen=rate
    elif np.size(rate)==np.size(t):
        rgen=rmax
    else:
        raise Exception('Error, the size of the rate is neither 1 nor equal to the size of the time')

    # generate spiketimes according to the maximum rate
    tspike=np.array([0.])
    while tspike[-1]<t[-1]:
        interval=np.array(random.expovariate(rgen))
        tspike=np.hstack([tspike,interval+tspike[-1]])

    tspike=np.delete(tspike, [0])
    tspike=np.delete(tspike, [np.size(tspike)-1])
    
    
    # generate the inhomogeneous poisson spikes
    if np.size(rate)!=1:
        
        spikebin=np.rint(tspike/dt)
        spikebin=spikebin.astype(int)
        
        rnorm=rate[spikebin]/rmax
        rtest=np.random.random_sample(np.size(rnorm))

        spikebin=spikebin[rtest<rnorm]
        tspike=spikebin*dt


    return tspike

# ------------------- stretched exponential distribution class
class str_exp(pyst.rv_continuous):
        
    def _pdf(self, x, p1, p2):
        return np.where(x<0, 0., 1. / p1/ ss.gamma(1. + 1. / p2) * np.exp(-(x / p1)**p2))

    def _cdf(self, x, p1, p2):
        return np.where(x<0, 0., (ss.gamma(1./p2) - ss.gamma(1./p2)* ss.gammaincc(1./p2, (x/p1)**p2)) / (p2 * ss.gamma(1.+1./p2)))

    
    def _rvs(self, p1, p2):
        rnd = np.random.rand(self._size[0])
        out = []
        x0 = 0.
        for nr in rnd:
            fun = lambda x : (self._cdf(x,p1,p2) - nr)**2           
            res_obj= sopt.minimize(fun,x0,method='SLSQP',options={'maxiter':1000})
            out.append(res_obj.x[0])
        return np.array(out)



# --------------------   Calculate moementa 1,2,3 of a distribution
def par2mom(par1,par2,distType='LogNormal'):
    if distType == 'LogNormal':
        mean = np.exp(par1 + 0.5*par2**2)
        sigma = np.exp(par1 + 0.5*par2**2)*np.sqrt(np.exp(par2**2) -1)
    elif distType == 'TruncNormal':
        h_1 = -par1/par2
        rho = pyst.norm.pdf(h_1) / (1-pyst.norm.cdf(h_1))
        mean = par1 + rho*par2
        sigma = par2 * np.sqrt(1-rho*(rho-h_1))
    elif distType == 'Exponential':
        scm = ncmom2csmom(mom_s_exp(par1,par2))
        mean = scm[0]
        sigma = scm[1]
    else:
        raise Exception('distType {} not accepted'.format(distType))

    return mean,sigma

def mom2par(mean,std,distType='LogNormal'):
    if distType=='LogNormal':
        par1=np.log(mean**2/np.sqrt(mean**2 + std**2))
        par2=np.sqrt(np.log(1 + std**2 / mean**2))

        return par1, par2

    elif distType=='TruncNormal':
        fun=lambda x : f_m2p(x,mean,std)
        x0=np.array([mean,std])
        bnds=(([0.,100.]),([0.,100.]))

        res_obj= sopt.minimize(fun,x0,method='SLSQP',bounds=bnds,options={'maxiter':1000}) 

        return res_obj.x[0], res_obj.x[1]

    elif distType == 'Exponential':
        fun = lambda x : f_m2p_exp(x,mean,std)

        x0=np.array([mean,std])
        bnds=(([0.,1000.]),([0.2,1000.]))

        res_obj= sopt.minimize(fun,x0,method='SLSQP',bounds=bnds,options={'maxiter':1000}) 

        return res_obj.x[0], res_obj.x[1]

    else:
        raise Exception('distType {} not accepted'.format(distType))

def f_m2p(x,mean_t,std_t):
    par1=x[0]
    par2=x[1]

    h_1=-par1/par2
    rho=pyst.norm.pdf(h_1) / (1-pyst.norm.cdf(h_1))
    mean=par1 + rho*par2
    std=par2*np.sqrt(1-rho*(rho-h_1))
    
    return ((mean-mean_t)**2 + (std-std_t)**2)

def f_m2p_exp(x,mean_t,std_t):
    par1=x[0]
    par2=x[1]

    mean, std = par2mom(par1,par2,distType='Exponential')

    return ((mean-mean_t)**2 + (std-std_t)**2)

def inputs2momenta_1D(nu,A,sigmaA,tau,tau1=0.):
    #vu=frequency, A=amplitude, sigmaA=std amplitude, tau=decay time of the inputs, tau1=rise time. it returns the mean and std of the process

    mu=A*nu*tau**2/(tau1+tau);
    stdev=np.sqrt(0.5*(A**2 + sigmaA**2)*nu*tau**3 / (tau+tau1) / (tau+2*tau1));

    return mu, stdev

def momenta2input_1D(mu,stdev,alpha,tau,tau1=0.):
    #you find rate and amplitude, starting from the momenta and the tau and amplitude variance parameter

    A=2*stdev**2 / (1+alpha**2) / mu * (tau+2*tau1)/tau;
    nu=(1+alpha**2)*mu**2 / 2 / stdev**2 * (tau1+tau)/(tau+2*tau1)/tau ;

    return A,nu


def inputs2momenta_full(nu,A,sigmaA,tau,tau1=0.,mom_n=np.arange(1,5),dist_type='LogNormal',weightType='Moments', n_moments=4):
    #vu=frequency, A=amplitude, sigmaA=std amplitude, tau=time constant of the inputs. it returns the moments of the process
    
    # HERE is integral of int_0^inf ((1-e^-t/T1)e^-t/T2)^n dt) unfortunately there is no nice equation describing it
    k_i_m=np.zeros(mom_n.shape)
    k_i_m[0]=tau**2/(tau1+tau)
    k_i_m[1]=tau**3/2/(tau1 + tau)/(2*tau1 + tau)
    k_i_m[2]=2*tau**4/3/(tau1 + tau)/(3*tau1+2*tau)/(3*tau1+tau)
    k_i_m[3]=3*tau**5/4/(tau1 + tau)/(4*tau1+3*tau)/(4*tau1+tau)/(2*tau1+tau)

    if dist_type=='LogNormal':
        if weightType=='Moments':# here we consider A, sigmaA as moments      
            mu=np.log((A**2)/np.sqrt(sigmaA**2+A**2))
            sigma=np.sqrt(np.log(sigmaA**2/A**2 +1))
        elif weightType=='Parameters':
            mu=A
            sigma=sigmaA
        else:
            raise Exception('weightType {} not accepted'.format(weightType))
        
        mom_mult=np.exp(mom_n*mu + 0.5*mom_n**2 * sigma**2)

    elif dist_type=='TruncNormal':
        if weightType=='Moments': # here we consider A, sigmaA as moments of the TruncNorm
            mu, sigma = mom2par(A,sigmaA,distType=dist_type)

        elif weightType=='Parameters':
            mu=A
            sigma=sigmaA
        else:
            raise Exception('weightType {} not accepted'.format(weightType))

        mom_mult=moments_truncgaussian2(mu,sigma)

    elif dist_type == 'Exponential':
        if weightType == 'Moments':
            par1, par2 = mom2par(A,sigmaA,distType=dist_type)
        
        elif weightType=='Parameters':
            par1=A
            par2=sigmaA
        else:
            raise Exception('weightType {} not accepted'.format(weightType))

        mom_mult=mom_s_exp(par1,par2)

    else:
        raise Exception('dist_type {} not accepted'.format(dist_type))
    
    cumulants=nu* mom_mult *k_i_m # these are the cumulants, or semi-invariants (1.5-2 rice) the central moments(see wikipedia) are the same, but the fourth one has a +3(sigma^4) term.
    c_mom=cumulants
    c_mom[mom_n==4]=c_mom[mom_n==4]+3*(c_mom[mom_n==2]**2)
        # now we standardize
    c_std=c_mom
    c_std[mom_n>2]=c_std[mom_n>2]/(c_mom[mom_n==2]**(mom_n[mom_n>2]/2))
    out=c_std
    out[mom_n==2]=np.sqrt(out[mom_n==2])
    out[mom_n==4]=out[mom_n==4]-3
    return out[:n_moments]

def empirical_moments(x,n_moments=4):
    ''' calculates the central moments <(I-<I>)^k>/std^k '''    
    mom_n=np.arange(0,4)
    mom=np.zeros(mom_n.shape)
    mom[0]=np.mean(x)
    mom[1]=np.std(x)
    
    for kk in mom_n[2:]:
        mom[kk]=pyst.moment(x,kk+1) / mom[1]**(kk+1)
  
    mom[3]=mom[3]-3
    return np.array(mom)[:n_moments]
 


      
def boots_moment(x,nsamples='def',nboots=1000):
    if nsamples=='def':
        nsamples=np.array(x.shape[0] / 10000)
        nsamples.astype(int)

    indeces=np.random.random_integers(0, x.shape[0]-1, [nboots,nsamples])    
    momenta=np.zeros([nboots,4])
        
    for ii in np.arange(0,nboots):
        momenta[ii,:]=empirical_moments(x[indeces[ii,:]])
        

    mom_mean=np.mean(momenta,axis=0)
    mom_std=np.std(momenta,axis=0)
    
    return mom_mean,mom_std



def power_spectrum_tau(f,A,tau,tau1):
    beta=(tau1/tau)
    omega=2*np.pi*f
    return A*tau**2  / ((1+beta)**2 + omega**2 *tau**2 * ((1+beta)**2 + beta**2) + tau**4 * beta**2 * omega**4)

def power_spectrum_full(f,A,stdA,vu,tau2,tau1):
    beta=np.absolute(tau1/tau2)
    omega=2*np.pi*f
    return 2*(A**2 + stdA**2)*vu*tau2**2  / ((1+beta)**2 + omega**2 *tau2**2 * ((1+beta)**2 + beta**2) + tau2**4 * beta**2 * omega**4)



# ------------------- TRUNCATED GAUSSIAN
def moments_truncgaussian(mu,sigma,myclip_a=0,myclip_b=10**10):

    if sigma>0:
        a_1, b_1 = (myclip_a - mu) / sigma, (myclip_b - mu) / sigma    
    # I find the central moments (not normalised!)
        
        c_mom_temp=pyst.truncnorm.stats(loc=mu,scale=sigma,a=a_1,b=b_1, moments='mvsk')
    
    else:
        c_mom_temp=np.array([mu,0.,0.,0.])
    
    c_mom=np.zeros(4)    
    c_mom[0]=c_mom_temp[0]
    c_mom[1]=c_mom_temp[1]
    c_mom[2]=c_mom_temp[2]*c_mom[1]**(3/2)
    c_mom[3]=(c_mom_temp[3]+3)*c_mom[1]**2

    # Here are the non central moments
    nc_mom=c_mom
    nc_mom[1]=c_mom[1] + c_mom[0]**2
    nc_mom[2]=c_mom[2] + 3*c_mom[1]*c_mom[0] + c_mom[0]**3
    nc_mom[3]=c_mom[3] + 4*c_mom[2]*c_mom[0] + 6*c_mom[1]*c_mom[0]**2 + c_mom[0]**4

    return nc_mom


def moments_truncgaussian2(mu,sigma):
    # c_mom are the central moments of the truncated gaussian
    if sigma >0:
        h_1=-mu/sigma
        rho=pyst.norm.pdf(h_1) / (1-pyst.norm.cdf(h_1))

        c_mom=np.zeros(4)
        c_mom[0]=mu + rho*sigma
        c_mom[1]=sigma**2 - sigma**2 * rho * (rho-h_1)
        c_mom[2]=sigma**3 * rho * ((rho-h_1)**2 + rho*(rho-h_1) -1)
        c_mom[3]=3*sigma**4 * (1- rho**2 * (rho-h_1)**2 - rho*(rho-h_1) - 1/3 * (h_1**2 * rho * (rho-h_1) -rho**2))

    else:
        c_mom=np.array([mu,0.,0.,0.])
    # nc_mom are the non-central moments of the truncated gaussian
    nc_mom=np.zeros(4)
    nc_mom[0]=c_mom[0]
    nc_mom[1]=c_mom[1] + c_mom[0]**2
    nc_mom[2]=c_mom[2] + 3*c_mom[1]*c_mom[0] + c_mom[0]**3
    nc_mom[3]=c_mom[3] + 4*c_mom[2]*c_mom[0] + 6*c_mom[1]*c_mom[0]**2 + c_mom[0]**4

    return nc_mom

def mom_s_exp(a,b):
    mom = np.zeros(4)
    for n in np.arange(1,5):
        mom[n-1] = ss.gamma((1.+n) / b) / (a**(-n) * b * ss.gamma(1 + 1./b))
    
    mom[np.isfinite(mom) == False] = 0.

    return mom

def ncmom2csmom(ncmom):
    csmom = np.zeros(4)
    csmom[0] = ncmom[0]
    csmom[1] = np.sqrt(ncmom[1]-ncmom[0]**2)
    csmom[2] = (2*ncmom[0]**3 -3*ncmom[0]*ncmom[1] + ncmom[2]) / csmom[1]**3
    csmom[3] = (-3*ncmom[0]**4 + 6*ncmom[0]**2 * ncmom[1] - 4 * ncmom[0]*ncmom[2] + ncmom[3])/csmom[1]**4 -3    
    return csmom


# --------------------  FILTERS

def general_filter(x,dt,cutoff,stopoff,Rp=0.2,attenuation=60.0,verbose=0):
    
    #------------------------------------------------
    # Create a FIR filter and apply it to x.
   #------------------------------------------------
    Fs=1/dt
    
    Wp=cutoff/Fs # cutoff
    Ws=stopoff/Fs # stopsfrequency
    
    As=attenuation
    
    bc,ac=sps.filter_design.iirdesign(Wp, Ws, Rp, As, ftype='ellip')

    if verbose>0:
        w, h = sps.freqz(bc,ac)
        plt.semilogy(w/(np.pi*dt), np.abs(h), 'b')
        plt.ylabel('Amplitude (dB)', color='b')
        plt.xlabel('Frequency (rad/sample)')
        
        ax=plt.gca()
        ax.set_xlim([0, 6000])
        ax.set_ylim([10**(-5), 1])
        plt.show()
    
    return sps.filtfilt(bc,ac,x)


    
