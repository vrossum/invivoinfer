from __future__ import division
import numpy as np
import logging

np.seterr(divide='warn')

import seaborn as sns

import scipy.stats as pyst
import scipy.signal as ssig
import scipy.optimize as sopt

import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import os
import json
import mc_functions as mcf
import vc_gen_class as vgc
import plot_scripts as plt_s
import imp
import pymc as pm

import pandas as pd

CONFIG_DIR = os.path.join("/".join(imp.find_module("invivoinfer")[1].split('/')[:-1]), 'config')
DATA_DIR = os.path.join("/".join(imp.find_module("invivoinfer")[1].split('/')[:-1]), 'data')


class VCInfer(object):
    # ---------------- CONSTRUCTOR
    def __init__(self, trace, dt, n_moments=4, params=None, figures_folder='figure_output', config=None):
        '''
        This class generates an object that contains a raw_trace and a timestep.
        In order to be constructed it needs the following parameters
        '''
        self.raw_trace = trace
        self.dt = dt
        self.descr_stat = {}
        self.descr_ps = {}
        self.n_moments = n_moments

        self.distType = None
        logging.info('Trace length: {} sec'.format(len(trace) * dt))

        if np.mean(self.raw_trace) < 0.:
            self.raw_trace = -self.raw_trace
            logging.info('We flip the trace to be positive, negative trace detected.')

        if config is None:
            self.params = self.default_params
        else:
            self.params = self.get_params_from_config(config)

        if params is not None:
            self.params.update(params)

        self.figure_dir = os.path.join(DATA_DIR, figures_folder)

        self.param_priors = dict()
        self.inference_results = {}
        self.figures = {}

    @property
    def default_params(self):
        config_file = os.path.join(CONFIG_DIR, 'config.json')
        with open(config_file, 'r') as fp:
            out = json.load(fp)
        return out

    def get_params_from_config(self, config):
        config_file = os.path.join(CONFIG_DIR, config)
        with open(config_file, 'r') as fp:
            out = json.load(fp)
        return out

    @property
    def weightType(self):
        # While for LogNormal the parameters denote the moments, for the other two distributions, they are
        # actually the parameters of the function

        if self.distType is None:
            return None
        elif self.distType == 'LogNormal':
            return 'Moments'
        elif self.distType == 'TruncNormal':
            return 'Parameters'
        elif self.distType == 'Exponential':
            return 'Parameters'
        return Exception('distType {} is not recognized'.format(self.distType))


        # @property
        # def init_param_priors(self):
        #     param_priors = {'tau1': np.zeros(2), 'tau2': np.zeros(2), 'constant': np.zeros(2), 'pw0': np.zeros(2),
        #                     'pw1': np.zeros(2), 'freq': np.zeros(2)}  # for each name, pos0: mean, pos1:std
        #
        #     for i in np.arange(self.n_moments):
        #         param_priors['obs_bias_{}'.format(i)] = [self.obs_bias['mean'][i], self.obs_bias['std'][i]]
        #
        #     return param_priors

        # ---------------- CALCULATING BASIC QUANTITIES

    def momenta(self):
        '''
        calculates mean, stdev, skew and kurtosis of the object 
        (no baseline corrections nor anything). 
        It's in self.descr_stat 
        '''

        self.descr_stat['mean'] = np.mean(self.raw_trace)
        self.descr_stat['std'] = np.std(self.raw_trace)
        self.descr_stat['skew'] = pyst.skew(self.raw_trace)
        self.descr_stat['kurtosis'] = pyst.kurtosis(self.raw_trace)

        logging.info(' ------  Summary of the statistics - RAW TRACE: \n {}'.format(self.descr_stat))

    def psd_calc(self, verbose=0, windowl=1., overlap_perc=0.75):
        '''
        it calculates the gross powerspectrum of the object
        it creates self.descr_ps
        '''

        self.psd_param = {'windowl': windowl, 'overlap_perc': overlap_perc}

        # using welch method
        window_bin = nextpow2(windowl / self.dt)
        overlap_bin = int(window_bin * overlap_perc)

        f, Pxx_den = ssig.welch(self.raw_trace, fs=1 / self.dt,
                                window='hanning', nperseg=window_bin, noverlap=overlap_bin, detrend='constant',
                                scaling='density')

        # estimate of the error (normal assumption)
        Pxx_den_std = Pxx_den * (window_bin / self.raw_trace.size * 11 / 9) ** 0.5

        self.descr_ps['psd_x'] = Pxx_den
        self.descr_ps['psd_x_std'] = Pxx_den_std
        self.descr_ps['psd_x_freq'] = f

    def estimate_taus(self, verbose=0, plot=0):
        '''
        from the powerspectrum, estimate taus and the constant Ps(nu)=constant*(tau2**2)/(.......)
        you need to run self.taus_init() beforehand
        '''
        taus_estimation_par = self.params['taus_estimation_par']

        logging.info('---------- Estimating taus from the powerspectrum ----------- ')

        # GOAL: to smoothen the uncertainty of the powerspectrum for the fit
        logging.info('* now smoothening the uncertainty onthe powerspectrum')
        logic_unc_fit = np.logical_and(self.descr_ps['psd_x_freq'] > taus_estimation_par['lim_fit_std'][0],
                                       self.descr_ps['psd_x_freq'] < taus_estimation_par['lim_fit_std'][1])
        fr_unc_fit = self.descr_ps['psd_x_freq'][logic_unc_fit]
        pow_std_unc_fit = self.descr_ps['psd_x_std'][logic_unc_fit]

        p_pot, dumb = sopt.curve_fit(exp_1, fr_unc_fit, pow_std_unc_fit, maxfev=fr_unc_fit.shape[0] * 1000)

        self.descr_ps['psd_x_std_smooth'] = exp_1(self.descr_ps['psd_x_freq'], p_pot[0], p_pot[1], p_pot[2])

        # using the datapoints only in the right range
        logging.info('* now preparing for the powerspectrum fit')
        logic_unc_mcmc_lim = np.logical_and(self.descr_ps['psd_x_freq'] > taus_estimation_par['lim_fit_psd'][0],
                                            self.descr_ps['psd_x_freq'] < taus_estimation_par['lim_fit_psd'][1])

        f = self.descr_ps['psd_x_freq'][logic_unc_mcmc_lim]
        data = self.descr_ps['psd_x'][logic_unc_mcmc_lim]
        data_err = self.descr_ps['psd_x_std_smooth'][logic_unc_mcmc_lim]

        # setting the starting points of the fit
        p0_tau1 = np.mean(taus_estimation_par['tau1_bound'])
        p0_tau2 = np.mean(taus_estimation_par['tau2_bound'])
        p0_const = data[0] / p0_tau2 ** 2
        p0 = [p0_const, p0_tau2, p0_tau1]

        fg_tau_res = 10 ** 20 * np.ones([taus_estimation_par['n_iter_tau']])
        fg_tau_data = {}
        fg_tau_data['constant'] = np.zeros(taus_estimation_par['n_iter_tau'])
        fg_tau_data['tau2'] = np.zeros(taus_estimation_par['n_iter_tau'])
        fg_tau_data['tau1'] = np.zeros(taus_estimation_par['n_iter_tau'])

        fun_tau = lambda x: f_tau_min(x, f, data_err, data)

        k = 0
        for kk in np.arange(taus_estimation_par['n_iter_tau']):
            logging.info('** Performing {} / {} optimisations'.format(kk + 1, taus_estimation_par['n_iter_tau']))
            p0_r = (np.random.rand(3) - 0.5) * p0 * 0.5 + p0

            if kk == 0:
                opt_par, dumb = sopt.curve_fit(mcf.power_spectrum_tau, f, data, p0=p0_r, sigma=data_err,
                                               maxfev=f.shape[0] * 1000)

            res_obj = sopt.minimize(fun_tau, x0=p0_r, method='Powell', options={'maxiter': 1000})  # Powell  'SLSQP'

            if (res_obj.success == True):
                fg_tau_data['constant'][kk] = res_obj.x[0]
                fg_tau_data['tau2'][kk] = res_obj.x[1]
                fg_tau_data['tau1'][kk] = res_obj.x[2]
                fg_tau_res[kk] = res_obj.fun
                k += 1

        logging.info('...and... {}/{} optimisations did converge'.format(k, kk + 1))

        fg_tau_data = pd.DataFrame(fg_tau_data)

        self.param_priors['tau1'] = [fg_tau_data['tau1'][np.argmin(fg_tau_res)], np.sqrt(dumb[2, 2])]
        self.param_priors['tau2'] = [fg_tau_data['tau2'][np.argmin(fg_tau_res)], np.sqrt(dumb[1, 1])]
        self.param_priors['constant'] = [fg_tau_data['constant'][np.argmin(fg_tau_res)], np.sqrt(dumb[0, 0])]

        logging.info('-------------- TAU ESTIMATION RESULTS -----------')
        logging.info('\nTau_1: {}+-{}, Tau_2: {}+-{}'.format(self.param_priors['tau1'][0], self.param_priors['tau1'][1],
                                                             self.param_priors['tau2'][0],
                                                             self.param_priors['tau2'][1]))
        logging.info('Constant: {}+-{}\n'.format(self.param_priors['constant'][0], self.param_priors['constant'][1]))

    def plot_tau_estimation(self, save_pdf=False):
        # Zoom on the low frequency part
        sns.set(style="ticks", context='paper')
        fig = plt.figure()
        plt.fill_between(self.descr_ps['psd_x_freq'], self.descr_ps['psd_x'] - self.descr_ps['psd_x_std'],
                         self.descr_ps['psd_x'] + self.descr_ps['psd_x_std'], facecolor='grey', edgecolor='grey')
        plt.plot(self.descr_ps['psd_x_freq'], self.descr_ps['psd_x'])

        plt.plot(self.descr_ps['psd_x_freq'],
                 mcf.power_spectrum_tau(self.descr_ps['psd_x_freq'], self.param_priors['constant'][0],
                                        self.param_priors['tau2'][0], self.param_priors['tau1'][0]), color='red',
                 label='fit')

        xlimit = 30
        ax = plt.gca()
        ax.set_xlim([0, xlimit])
        ax.set_ylim([np.min(
            self.descr_ps['psd_x'][self.descr_ps['psd_x_freq'] < xlimit] - self.descr_ps['psd_x_std'][
                self.descr_ps['psd_x_freq'] < xlimit]),
            np.max(self.descr_ps['psd_x'] + self.descr_ps['psd_x_std'])])
        ax.set_yscale("log", nonposx='clip')

        sns.despine()
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V$^2$/Hz]')
        plt.tight_layout(pad=2)
        if save_pdf:
            plt_s.save_plot(os.path.join(self.figure_dir, 'psp_lf_zoom.pdf'), fig_size=[8, 5], file_format='pdf')

        self.figures['psp_lf_zoom'] = fig

        # All powerspectrum
        fig = plt.figure()
        sns.set(style="ticks", context='paper')

        plt.fill_between(self.descr_ps['psd_x_freq'], self.descr_ps['psd_x'] - self.descr_ps['psd_x_std'],
                         self.descr_ps['psd_x'] + self.descr_ps['psd_x_std'], facecolor='grey', edgecolor='grey')
        plt.plot(self.descr_ps['psd_x_freq'], self.descr_ps['psd_x'])

        plt.plot(self.descr_ps['psd_x_freq'],
                 mcf.power_spectrum_tau(self.descr_ps['psd_x_freq'], self.param_priors['constant'][0],
                                        self.param_priors['tau2'][0], self.param_priors['tau1'][0]), color='red',
                 label='fit')

        xlimit = 1000
        ax = plt.gca()
        ax.set_xlim([0, xlimit])
        ax.set_ylim([np.min(
            self.descr_ps['psd_x'][self.descr_ps['psd_x_freq'] < xlimit] - self.descr_ps['psd_x_std'][
                self.descr_ps['psd_x_freq'] < xlimit]),
            np.max(self.descr_ps['psd_x'] + self.descr_ps['psd_x_std'])])
        ax.set_yscale("log", nonposx='clip')

        sns.despine()
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V$^2$/Hz]')
        plt.tight_layout(pad=2)

        if save_pdf:
            plt_s.save_plot(os.path.join(self.figure_dir, 'psp_all.pdf'), fig_size=[8, 5], file_format='pdf')

        self.figures['psp_all'] = fig

    # --------------------- SETTING PRIORS ON DATA MEASUREMENT BIASES, namely P(Dobs|Dtrue)

    def reset_obs_unc(self):
        logging.info('*** Resetting observations bias and uncertainties')
        self.obs_bias = {'mean': np.zeros(self.n_moments),
                         'std': 1. * 10 ** (-1) * np.ones(self.n_moments)}  # from position 0..4: mean,std,skew,kurt

        logging.info('Obs uncertainties: {}+-{}'.format(self.obs_bias['mean'], self.obs_bias['std']))

    def set_obs_unc_mean(self):
        '''
        it sets the bias and uncertainty on the recording (namely the baseline)
        '''
        base_line_force = self.params['baseline_corr']
        if base_line_force['ToUse'] == True:
            logging.info('\n ---------------  Assigning bias and uncertainty of the mean -----------')
            logging.info('Baseline: {}+-{}\n'.format(base_line_force['average'], base_line_force['uncertainty']))

            self.obs_bias['mean'][0] = base_line_force['average']
            self.obs_bias['std'][0] = base_line_force['uncertainty']

    def plot_mean_offset(self, save_pdf=False):
        base_line_force = self.params['baseline_corr']

        fig = plt.figure()
        t_plot = np.linspace(self.dt, self.dt * self.raw_trace.shape[0], self.raw_trace.shape[0])
        plt.plot(t_plot, self.raw_trace)

        plt.plot(t_plot, np.ones(len(t_plot)) * base_line_force['average'])
        plt.plot(t_plot,
                 np.ones(len(t_plot)) * (base_line_force['average'] - base_line_force['uncertainty']))
        plt.plot(t_plot,
                 np.ones(len(t_plot)) * (base_line_force['average'] + base_line_force['uncertainty']))
        plt.ylim([0, 2 * base_line_force['average']])
        plt.tight_layout(pad=2)

        if save_pdf:
            plt_s.save_plot(os.path.join(self.figure_dir, 'baseline.pdf'), fig_size=[8, 5], file_format='pdf')

        self.figures['baseline'] = fig

    def set_obs_unc_std(self):
        '''
        it sets the high freq noise bias and uncertainty on the std (generally you can measure it easily)
        it estimates the contribution of low freq oscillations to the powerspectrum (given the cutoff threshold)
        VERBOSE 1 print, 2 plot
        '''
        hf_std_corr = self.params['hf_std_corr']
        lf_std_corr = self.params['lf_std_corr']

        logging.info('\n -------- Measuring the bias of the std --------')

        # HIGH FREQUENCY
        if hf_std_corr['ToUse'] == True:
            hfnoise_mean = hf_std_corr['average']
            hfnoise_std = hf_std_corr['uncertainty']
        else:
            hfnoise_mean = 0
            hfnoise_std = 0
        logging.info('High frequency noise: {}+-{}'.format(hfnoise_mean, hfnoise_std))

        # LOW FREQUENCY
        if lf_std_corr['ToUse'] == True:
            # predicted_std is the area under the fitted curve used to get the tau
            predicted_std = np.sqrt(self.param_priors['constant'][0] * self.param_priors['tau2'][0] ** 3 / 4 / (
                self.param_priors['tau2'][0] + self.param_priors['tau1'][0]) / (
                                        (self.param_priors['tau2'][0] + 2 * self.param_priors['tau1'][0])))

            if self.descr_stat['std'] ** 2 - predicted_std ** 2 > 0:
                lfnoise_mean = np.sqrt(self.descr_stat['std'] ** 2 - predicted_std ** 2)
            else:
                lfnoise_mean = 0
            lfnoise_std = 0
        else:
            lfnoise_mean = 0
            lfnoise_std = 0
        logging.info('Low frequency noise: {}+-{}'.format(lfnoise_mean, lfnoise_std))

        if hf_std_corr['ToUse'] == True or lf_std_corr['ToUse'] == True:
            self.obs_bias['mean'][1] = self.descr_stat['std'] - np.sqrt(
                self.descr_stat['std'] ** 2 - (hfnoise_mean ** 2 + lfnoise_mean ** 2))
            self.obs_bias['std'][1] = np.sqrt(hfnoise_std ** 2 + lfnoise_std ** 2)

            # HACK TO IMPROVE KURT AND SKEW:
            # self.obs_bias['mean'][2] = -self.obs_bias['mean'][1] / 10
            # self.obs_bias['mean'][3] = -self.obs_bias['mean'][1] / 10

        logging.info('Total noise contribution to std:{}+-{}'.format(self.obs_bias['mean'][1], self.obs_bias['std'][1]))

    def first_guess(self, distType='LogNormal'):
        '''
        Given the taus estimated, it calculates the parameters A, stdA, freq from a LS fit.
        It caluclates also their uncertainties by starting the fit from different starting points
        The results are put in self.param_priors
        '''
        self.distType = distType
        first_guess_par = self.params['first_guess_par'][distType]

        pw0_bound = first_guess_par['pw0_bound']
        pw1_bound = first_guess_par['pw1_bound']
        freq_bound = first_guess_par['freq_bound']
        n_iter_par = first_guess_par['n_iter_par']

        logging.info('\n --START OF THE FIRST GUESS OPTIMIZATION.....')

        first_guess_s = {}
        # preparing the minimization

        t1_fix = self.param_priors['tau1'][0]
        t2_fix = self.param_priors['tau2'][0]
        if self.n_moments == 4:
            y_target = np.array([self.descr_stat['mean'] - self.obs_bias['mean'][0],
                                 self.descr_stat['std'] - self.obs_bias['mean'][1],
                                 self.descr_stat['skew'] - self.obs_bias['mean'][2],
                                 self.descr_stat['kurtosis'] - self.obs_bias['mean'][3]])
        else:
            y_target = np.array([self.descr_stat['mean'] - self.obs_bias['mean'][0],
                                 self.descr_stat['std'] - self.obs_bias['mean'][1],
                                 self.descr_stat['skew'] - self.obs_bias['mean'][2],
                                 ])

        fun = lambda x: f_like(x, t1_fix, t2_fix, distType, y_target, n_moments=self.n_moments)
        first_guess_s['pw0'] = np.zeros(n_iter_par)
        first_guess_s['pw1'] = np.zeros(n_iter_par)
        first_guess_s['freq'] = np.zeros(n_iter_par)
        residuals = 10 ** 5 * np.ones(n_iter_par)  # This is the default residuals if fail to converge (very high!)

        bnds = ((pw0_bound), (pw1_bound), (freq_bound))
        k = 0
        for jj in np.arange(n_iter_par):
            x0 = ((np.random.rand() - 0.5) * np.diff(pw0_bound) + np.mean(pw0_bound),
                  (np.random.rand() - 0.5) * np.diff(pw1_bound) + np.mean(pw1_bound),
                  (np.random.rand() - 0.5) * np.diff(freq_bound) + np.mean(freq_bound))

            res_obj = sopt.minimize(fun, x0, method='SLSQP', bounds=bnds,
                                    options={'maxiter': 1000})  # fitting the first four momenta

            if (res_obj.success == True):
                first_guess_s['pw0'][jj] = res_obj.x[0]
                first_guess_s['pw1'][jj] = res_obj.x[1]
                first_guess_s['freq'][jj] = res_obj.x[2]
                residuals[jj] = res_obj.fun
                k = k + 1

        logging.info('...and... {}/{} optimisations did converge'.format(k, jj + 1))

        for i, val in enumerate(first_guess_s):
            m_val = first_guess_s[val][np.argmin(residuals)]
            # The next is just a reasonable estimate of the variability of the first guess estimation
            unc_val = np.std(first_guess_s[val][first_guess_s[val] != 0])
            self.param_priors[val] = np.array([m_val, unc_val])

        logging.info('Values for the {} of the {} weights distribution:'.format(self.weightType, distType))
        logging.info('PW0: {}+-{}, PW1: {}+-{}, freq: {}+-{}\n'.format(self.param_priors['pw0'][0],
                                                                       self.param_priors['pw0'][1],
                                                                       self.param_priors['pw1'][0],
                                                                       self.param_priors['pw1'][1],
                                                                       self.param_priors['freq'][0],
                                                                       self.param_priors['freq'][1]))
        if self.weightType == 'Parameters':
            m_w, s_w = mcf.par2mom(self.param_priors['pw0'][0], self.param_priors['pw1'][0], distType=distType)
        else:
            m_w, s_w = self.param_priors['pw0'][0], self.param_priors['pw1'][0]
        logging.info('Moments of the weights distibution:')
        logging.info('Mean: {}, Std: {}\n'.format(m_w, s_w))

        logging.info('First guess theoretical moments: {}'.format(
            mcf.inputs2momenta_full(self.param_priors['freq'][0], self.param_priors['pw0'][0],
                                    self.param_priors['pw1'][0], self.param_priors['tau2'][0],
                                    self.param_priors['tau1'][0], np.array([1, 2, 3, 4]), dist_type=distType,
                                    weightType=self.weightType))
        )
        logging.info('Target experimental moments: {}'.format(y_target))
        logging.info('\n --END OF THE FIRST GUESS OPTIMIZATION ..... \n')

    def optimize_likelihood(self):
        ''' 
        To optimize the likelihood I run nsamples of the first guess
        The output is the std/mean of the momenta and tau
        '''
        logging.info('\n -- START OF THE ESTIMATION OF THE LIKELIHOOD FUNCTION ...')
        sk_ku_corr = self.params['sk_ku_corr']
        likelihood_par = self.params['likelihood_par']
        kernel_par = self.params['kernel_par']

        sim_param = {'time': self.dt * self.raw_trace.shape[0], 'dt': self.dt}

        if sk_ku_corr['ToUse'] == True:
            sk_ku_corr['lf_noise'] = np.sqrt(
                self.obs_bias['mean'][1] * (self.descr_stat['std'] - self.obs_bias['mean'][1]))
        else:
            sk_ku_corr['lf_noise'] = 0

        kde_ll, mom_sim, mom_bias = self.likelihood_est_f(likelihood_par['nsamples'],
                                                          self.param_priors,
                                                          sim_param,
                                                          self.distType, self.weightType,
                                                          kernel_par=kernel_par,
                                                          sk_ku_corr=sk_ku_corr,
                                                          n_moments=self.n_moments)

        self.likelihood_estim = {'kde_ll': kde_ll, 'moments_sim': mom_sim}

        if sk_ku_corr['ToUse'] == True:
            # This are the uncertainties, which are bigger now, as shown by the variability of the simulation
            self.obs_bias['std'] = np.sqrt(
                self.obs_bias['std'] ** 2 + mom_bias['std'] ** 2)

            # Since bias of mean and std have been already been accounted for, we correct onlt skew and kurt
            self.obs_bias['mean'][2] = self.obs_bias['mean'][2] + mom_bias['mean'][2]  # bias of the skew
            if self.n_moments == 4:
                self.obs_bias['mean'][3] = self.obs_bias['mean'][3] + mom_bias['mean'][3]  # bias of the kurtosis

        logging.info('New biases: {}+-{}'.format(self.obs_bias['mean'], self.obs_bias['std']))

        logging.info('\n -- END OF OPTIMIZATION OF THE LIKELIHOOD...')

    def likelihood_est_f(self, nsamples, param_priors, sim_param, distType, weightType,
                         kernel_par={'Type': 'exponential', 'BW': 1},
                         sk_ku_corr={'ToUse': False, 'lf_noise': 0},
                         n_moments=4
                         ):

        # --------------------       SET UP THE INPUTS TO THE SIMULATION

        Syn_exc_par = {}
        Syn_inh_par = {}
        Noise_par = {}
        Noise_par['ampli'] = 0.  # std of noise amplitude in pA

        Noise_par['LF_ampli'] = sk_ku_corr['lf_noise']
        Noise_par['LF_cutoff'] = 2.5

        Fr_par = {}
        Init_par = {}
        Sim_par = {}

        # --------- Excitatory
        Syn_exc_par['tau1'] = param_priors['tau1'][0]  # seconds
        Syn_exc_par['tau2'] = param_priors['tau2'][0]
        Syn_exc_par['A'] = param_priors['pw0'][0]  # in pA
        Syn_exc_par['stdA'] = param_priors['pw1'][0]
        Syn_exc_par['WeightDist'] = distType  # TruncNormal/LogNormal/Exponential
        Syn_exc_par['WeightType'] = weightType

        Syn_exc_par['freq'] = param_priors['freq'][0]

        Init_par['Ioffset'] = 0.

        Sim_par['duration'] = sim_param['time']
        Sim_par['dt'] = sim_param['dt']

        # Calculate theoretical moments (without LF noise!)
        moments = mcf.inputs2momenta_full(Syn_exc_par['freq'], Syn_exc_par['A'], Syn_exc_par['stdA'],
                                          Syn_exc_par['tau2'],
                                          Syn_exc_par['tau1'], dist_type=Syn_exc_par['WeightDist'],
                                          weightType=Syn_exc_par['WeightType'])
        logging.info('**********************  Likelihood function shape calculation - running...')
        logging.info('{} of the {} weight distribution'.format(Syn_exc_par['WeightType'], Syn_exc_par['WeightDist']))
        logging.info('pw0: {}, pw1: {}, freq: {}, tau1: {}, tau2: {}'.format(
            Syn_exc_par['A'], Syn_exc_par['stdA'], Syn_exc_par['freq'], Syn_exc_par['tau1'], Syn_exc_par['tau2'])
        )
        logging.info('...Starting simulations to find the uncertainties on the likelihood...')

        # initialize the arrays
        momenta_sample = np.zeros([nsamples, n_moments])
        momenta_sample_noise = np.zeros([nsamples, n_moments])

        for kk in np.arange(nsamples):

            v1 = vgc.VCGeneration(Syn_exc_par, Syn_inh_par, Noise_par, Fr_par, Init_par, Sim_par)
            v1.reset()
            v1.run()
            momenta_sample[kk, :] = mcf.empirical_moments(v1.I, n_moments=n_moments)
            if sk_ku_corr['ToUse'] == True:
                if Noise_par['LF_ampli'] > 0:
                    v1.add_OU_noise(0, Noise_par['LF_ampli'], Noise_par['LF_cutoff'])
                    momenta_sample_noise[kk, :] = mcf.empirical_moments(v1.I, n_moments=n_moments)
            else:
                momenta_sample_noise[kk, :] = momenta_sample[kk, :]

            if kk % 10 == 0:
                logging.info('Simulated {} / {}'.format(kk, nsamples))
            else:
                logging.debug('Simulated {} / {}'.format(kk, nsamples))

        if n_moments == 4:
            col_names = ['mean', 'std', 'skew', 'kurtosis']
        else:
            col_names = ['mean', 'std', 'skew']

        self.likelihood_samples = pd.DataFrame(momenta_sample, columns=col_names)

        mom_sim = {'mean': np.mean(momenta_sample, axis=0), 'std': np.std(momenta_sample, axis=0)}
        momenta_standard = (momenta_sample - mom_sim['mean']) / mom_sim['std']

        kde_ll = KernelDensity(kernel=kernel_par['Type'], bandwidth=kernel_par['BW'])
        kde_ll.fit(momenta_standard)

        # biases calculation
        momenta_bias_sample = momenta_sample_noise - momenta_sample
        mom_bias = {'mean': np.mean(momenta_bias_sample, axis=0), 'std': np.std(momenta_bias_sample, axis=0)}

        if n_moments == 4:
            logging.info('Theoretical moments. mean: {}, std: {}, skew: {}, kurt: {}'.format(
                moments[0], moments[1], moments[2], moments[3])
            )
            logging.info(
                'Measured moments. mean: {}+{}, std: {}+-{}, skew: {}+-{}, kurt: {}+-{}'.format(mom_sim['mean'][0],
                                                                                                mom_sim['std'][0],
                                                                                                mom_sim['mean'][1],
                                                                                                mom_sim['std'][1],
                                                                                                mom_sim['mean'][2],
                                                                                                mom_sim['std'][2],
                                                                                                mom_sim['mean'][3],
                                                                                                mom_sim['std'][3])
                )
            logging.info(
                'Measured bias. mean: {}+{}, std: {}+-{}, skew: {}+-{}, kurt: {}+-{}'.format(mom_bias['mean'][0],
                                                                                             mom_bias['std'][0],
                                                                                             mom_bias['mean'][1],
                                                                                             mom_bias['std'][1],
                                                                                             mom_bias['mean'][2],
                                                                                             mom_bias['std'][2],
                                                                                             mom_bias['mean'][3],
                                                                                             mom_bias['std'][3])
                )

        else:
            logging.info('Theoretical moments. mean: {}, std: {}, skew: {}'.format(
                moments[0], moments[1], moments[2])
            )
            logging.info('Measured moments. mean: {}+{}, std: {}+-{}, skew: {}+-{}'.format(mom_sim['mean'][0],
                                                                                           mom_sim['std'][0],
                                                                                           mom_sim['mean'][1],
                                                                                           mom_sim['std'][1],
                                                                                           mom_sim['mean'][2],
                                                                                           mom_sim['std'][2],
                                                                                           )
                         )
            logging.info('Measured bias. mean: {}+{}, std: {}+-{}, skew: {}+-{}'.format(mom_bias['mean'][0],
                                                                                        mom_bias['std'][0],
                                                                                        mom_bias['mean'][1],
                                                                                        mom_bias['std'][1],
                                                                                        mom_bias['mean'][2],
                                                                                        mom_bias['std'][2],
                                                                                        )
                         )

        return kde_ll, mom_sim, mom_bias

    def plot_likelihood(self, save_pdf=False):

        fig = plt.figure()
        g = sns.PairGrid(self.likelihood_samples)
        g.map_upper(plt.scatter)
        g.map_lower(sns.kdeplot, cmap="Blues_d")
        g.map_diag(sns.kdeplot, lw=3, legend=False)
        plt.tight_layout(pad=2)

        if save_pdf:
            plt_s.save_plot(os.path.join(self.figure_dir, 'likelihood.pdf'), fig_size=[8, 5], file_format='pdf')

        self.figures['likelihood'] = fig

    def create_model(self):
        '''
        '''

        # SETTING UP THE PRIORS
        model_param = self.params['prior_par'][self.distType]
        freq_prior = model_param['freq']
        if freq_prior[0] == 'Uniform':
            freq = pm.Uniform("freq", freq_prior[1], freq_prior[2])
        elif freq_prior[0] == 'Normal':
            freq = pm.TruncatedNormal("freq", mu=freq_prior[1], tau=1 / freq_prior[2] ** 2, a=freq_prior[3],
                                      b=freq_prior[4])
        else:
            raise Exception('Prior for pw0 must be in [Uniform, Normal]')

        pw0_prior = model_param['pw0']
        if pw0_prior[0] == 'Uniform':
            pw0 = pm.Uniform("pw0", pw0_prior[1], pw0_prior[2])
        elif pw0_prior[0] == 'Normal':
            pw0 = pm.TruncatedNormal("pw0", mu=pw0_prior[1], tau=1 / pw0_prior[2] ** 2, a=pw0_prior[3], b=pw0_prior[4])
        else:
            raise Exception('Prior for pw1 must be in [Uniform, Normal]')

        pw1 = model_param['pw1']
        if pw1[0] == 'Uniform':
            pw1 = pm.Uniform("pw1", pw1[1], pw1[2])
        elif pw1[0] == 'Normal':
            pw1 = pm.TruncatedNormal("pw1", mu=pw0, tau=1 / (pw1[2]) ** 2, a=pw1[2], b=pw1[4])
        elif pw1[0] == 'Exponential':
            pw1 = pm.Exponential("pw1", beta=1. / (pw1[1]))
        else:
            raise Exception('Prior for pw0 must be in [Uniform, Normal, Exponential]')

        tau2 = self.param_priors['tau2'][0]
        tau1 = self.param_priors['tau1'][0]

        # ------ SETTING UP THE LIKELIHOOD INGREDIENTS

        # ---- P(mean(Dtrue)|model)

        @pm.deterministic
        def mean_moments(freq=freq, A=pw0, stdA=pw1, tau2=tau2, tau1=tau1, dist_type=self.distType,
                         n_moments=self.n_moments):  # analytical calculation of the mean of first 4 moments of the distribuion, given the parameters (extending the mom_n array makes predictions of the other moments.
            mom_n = np.arange(1, 4 + 1)
            k_i_m = np.zeros(mom_n.shape)
            k_i_m[0] = tau2 ** 2 / (tau1 + tau2)
            k_i_m[1] = tau2 ** 3 / 2 / (tau1 + tau2) / (2 * tau1 + tau2)
            k_i_m[2] = 2 * tau2 ** 4 / 3 / (tau1 + tau2) / (3 * tau1 + 2 * tau2) / (3 * tau1 + tau2)
            k_i_m[3] = 3 * tau2 ** 5 / 4 / (tau1 + tau2) / (4 * tau1 + 3 * tau2) / (4 * tau1 + tau2) / (2 * tau1 + tau2)

            if dist_type == 'LogNormal':
                mu = np.log((A ** 2) / np.sqrt(stdA ** 2 + A ** 2))
                sigma = np.sqrt(np.log(stdA ** 2 / A ** 2 + 1))

                mom_mult = np.exp(mom_n * mu + 0.5 * mom_n ** 2 * sigma ** 2)

            elif dist_type == 'TruncNormal':
                mom_mult = mcf.moments_truncgaussian2(A, stdA)

            elif dist_type == 'Exponential':
                stdA = np.amin([stdA, 12.])
                mom_mult = mcf.mom_s_exp(A, stdA)

            else:
                raise Exception('Wrong type of distribution! {}'.format(dist_type))

            cumulants = freq * mom_mult * k_i_m  # these are the cumulants, or semi-invariants (1.5-2 rice) the central moments(see wikipedia) are the same, but the fourth one has a +3(sigma^4) term.

            c_mom = cumulants;
            c_mom[mom_n == 4] = c_mom[mom_n == 4] + 3 * (c_mom[mom_n == 2] ** 2)
            # now we standardize
            c_std = c_mom
            c_std[mom_n > 2] = c_std[mom_n > 2] / (c_mom[mom_n == 2] ** (mom_n[mom_n > 2] / 2))
            out = c_std
            out[mom_n == 2] = np.sqrt(out[mom_n == 2])
            out[mom_n == 4] = out[mom_n == 4] - 3

            out[np.isfinite(out) == False] = 0.

            return out[:n_moments]

        # ---- P(Dobs|Dtrue) probab of observed data given th true ones (corrupted by biases and uncertainties due to baseline, hf/lf noise

        self.obs_bias['std'][
            self.obs_bias['std'] == 0] = 10 ** -2  # Just in case we have some 0 std! The likelihood would shoot

        obs_bias_0 = pm.Normal('obs_bias_0', mu=self.obs_bias['mean'][0], tau=1 / (self.obs_bias['std'][0]) ** 2)
        obs_bias_1 = pm.Normal('obs_bias_1', mu=self.obs_bias['mean'][1], tau=1 / (self.obs_bias['std'][1]) ** 2)
        obs_bias_2 = pm.Normal('obs_bias_2', mu=self.obs_bias['mean'][2], tau=1 / (self.obs_bias['std'][2]) ** 2)
        if self.n_moments == 4:
            obs_bias_3 = pm.Normal('obs_bias_3', mu=self.obs_bias['mean'][3], tau=1 / (self.obs_bias['std'][3]) ** 2)

        logging.info('OBSERVATION BIASES ----   P(Dmeas|Dtrue)')
        for i, val in enumerate(self.obs_bias['mean']):
            logging.info('{} +- {}'.format(self.obs_bias['mean'][i], self.obs_bias['std'][i]))

        for i in np.arange(self.n_moments):  # PASS the obs_bias to the parameters values
            self.param_priors['obs_bias_{}'.format(i)] = [self.obs_bias['mean'][i], self.obs_bias['std'][i]]

        if self.n_moments == 4:
            @pm.deterministic
            def obs_data(mean_moments=mean_moments,
                         obs_bias_0=obs_bias_0,
                         obs_bias_1=obs_bias_1,
                         obs_bias_2=obs_bias_2,
                         obs_bias_3=obs_bias_3):
                obs_bias = np.array([obs_bias_0, obs_bias_1, obs_bias_2, obs_bias_3])
                return mean_moments + obs_bias
        else:
            @pm.deterministic
            def obs_data(mean_moments=mean_moments,
                         obs_bias_0=obs_bias_0,
                         obs_bias_1=obs_bias_1,
                         obs_bias_2=obs_bias_2,
                         ):
                obs_bias = np.array([obs_bias_0, obs_bias_1, obs_bias_2])
                return mean_moments + obs_bias

        # -------------- DATA OBSERVED
        if self.n_moments == 4:
            data_observed = np.array(
                [self.descr_stat['mean'], self.descr_stat['std'], self.descr_stat['skew'], self.descr_stat['kurtosis']])
        else:
            data_observed = np.array(
                [self.descr_stat['mean'], self.descr_stat['std'], self.descr_stat['skew']])

        logging.info('MEASURED DATA {}'.format(data_observed))

        # -------------- LIKELIHOOD FUNCTION P(Dobs|param)
        @pm.stochastic(observed=True)
        def obs_like(value=data_observed, obs_data=obs_data,
                     ll_kernel=self.likelihood_estim['kde_ll'],
                     ll_old_std=self.likelihood_estim['moments_sim']['std'],
                     ):

            target_new_std = (
                             value - obs_data) / ll_old_std  # I normalise the differences in order to use the standardised kernel

            if np.isfinite(ll_kernel.score(target_new_std.reshape(-1, len(target_new_std)))) == False:
                return -10. ** 10.
            else:
                return ll_kernel.score(target_new_std.reshape(-1, len(target_new_std)))

        if self.n_moments == 4:
            self.model = pm.Model(
                [obs_like, pw0, pw1, freq, tau2, tau1, obs_bias_0, obs_bias_1, obs_bias_2, obs_bias_3])
        else:
            self.model = pm.Model([obs_like, pw0, pw1, freq, tau2, tau1, obs_bias_0, obs_bias_1, obs_bias_2])

    def run_sampler(self, sampler_type='mh'):
        '''
        run the sampler on the model created.
        'sampler_type': 'mh' for metropolis hastings (only this supported so far)
        '''

        # --------------------    CALCULATING THE MAP

        self.inference_results['map'] = MAP_estimate(self.model, self.param_priors, self.obs_bias,
                                                     n_iter=self.params['map_par']['n_samples'],
                                                     start_std=self.params['map_par']['start_std'],
                                                     distType=self.distType)

        # ------ PRINT MAP PREDICTION
        if self.n_moments == 4:
            y_target = np.array([self.descr_stat['mean'] - self.obs_bias['mean'][0],
                                 self.descr_stat['std'] - self.obs_bias['mean'][1],
                                 self.descr_stat['skew'] - self.obs_bias['mean'][2],
                                 self.descr_stat['kurtosis'] - self.obs_bias['mean'][3]])
        else:
            y_target = np.array([self.descr_stat['mean'] - self.obs_bias['mean'][0],
                                 self.descr_stat['std'] - self.obs_bias['mean'][1],
                                 self.descr_stat['skew'] - self.obs_bias['mean'][2]]
                                )

        logging.info('Measured (corrected) moments: {}'.format(y_target))
        logging.info('MAP predicted moments: {}'.format(
            mcf.inputs2momenta_full(self.inference_results['map']['parameters']['freq'].values,
                                    self.inference_results['map']['parameters']['pw0'].values,
                                    self.inference_results['map']['parameters']['pw1'].values,
                                    self.param_priors['tau2'][0],
                                    self.param_priors['tau1'][0],
                                    np.array([1, 2, 3, 4]),
                                    dist_type=self.distType,
                                    weightType=self.weightType,
                                    n_moments=self.n_moments)))

        # --------------------    RUN THE SAMPLER
        sampler_param = self.params['mc_par']
        if sampler_type == 'mh':
            self.inference_results['mh'] = run_mh(self.model, self.inference_results['map'],
                                                  sampler_param['mh'], distType=self.distType,
                                                  )

    def plot_mc_results(self, save_pdf=False):
        figures = _plot_mc_results(self.inference_results['map'], self.inference_results['mh'], save_pdf, self.figure_dir)

        self.figures['mh'] = figures['mh']



# --------------- plot summary results
def _plot_mc_results(map_data, mh_data, save_pdf=False, figure_dir=None):

    figures = {}

    logging.info('START PLOTTING FIGURES')

    fig = plt.figure()
    g = sns.PairGrid(mh_data['weights'])
    g.map_upper(plt.scatter)
    g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.map_diag(sns.kdeplot, lw=3)

    plt.tight_layout(pad=2)

    if save_pdf:
        plt_s.save_plot(os.path.join(figure_dir, 'mh_plot.pdf'), fig_size=[8, 5], file_format='pdf')

    figures['mh'] = fig

    return figures

# --------------- LIKELIHOOD ESTIMATION FUNCTION

# ----------------------   MAP

def MAP_estimate(model, param_priors, obs_bias, n_iter=50, start_std=0.2, distType='LogNormal'):
    # ------------------    FIND MAP  given a model (you need to have calculated the first guess before!
    logging.info('\n\nCalculating the MAP')
    map_ = pm.MAP(model)
    map_sum = {}
    map_bic = np.zeros(np.size(np.arange((n_iter) * 2)))
    map_aic = np.zeros(np.size(np.arange((n_iter) * 2)))

    logging.info('Starting condition close to first guess')

    for l in np.arange(n_iter):
        for i, var in enumerate(model.stochastics):
            var.value = param_priors[str(var)][0] + start_std * param_priors[str(var)][0] * np.random.randn()
            # I start looking for the MAP from the first-guess results!
            if l == 0:
                map_sum[str(var)] = []
        try:
            map_.fit()
            map_aic[l] = map_.AIC
            map_bic[l] = map_.BIC

        except Exception:
            print Exception
            pass
        for i, var in enumerate(model.stochastics):
            map_sum[str(var)] = np.hstack((map_sum[str(var)], var.value))

    logging.info('Starting condition uniform random')

    for l in np.arange(n_iter, np.size(np.arange((n_iter) * 2))):
        model.draw_from_prior()
        try:
            map_.fit()
            map_aic[l] = map_.AIC
            map_bic[l] = map_.BIC

        except Exception:
            print Exception
            pass
        for i, var in enumerate(model.stochastics):
            map_sum[str(var)] = np.hstack((map_sum[str(var)], var.value))

    map_sum = pd.DataFrame.from_dict(map_sum)
    if distType == 'TruncNormal':
        mu_sum, sigma_sum = mcf.par2mom(map_sum['pw0'].values, map_sum['pw1'].values, distType=distType)
        map_sum_w = pd.DataFrame(np.array([mu_sum, sigma_sum]).T, columns=['A', 'stdA'])
        map_sum_w['freq'] = map_sum['freq']
    elif distType == 'Exponential':
        mu_sum = np.zeros(np.shape(map_sum['pw0'].values))
        sigma_sum = np.zeros(np.shape(map_sum['pw0'].values))
        for i, val in enumerate(map_sum['pw0'].values):
            mu_sum[i], sigma_sum[i] = mcf.par2mom(map_sum['pw0'].values[i], map_sum['pw1'].values[i], distType=distType)

        map_sum_w = pd.DataFrame(np.array([mu_sum, sigma_sum]).T, columns=['A', 'stdA'])
        map_sum_w['freq'] = map_sum['freq']
    elif distType == 'LogNormal':
        map_sum_w = map_sum[['pw0', 'pw1', 'freq']]
        map_sum_w.columns = ['A', 'stdA', 'freq']
    else:
        raise Exception('distType {} not supported'.format(distType))

    mb_value = np.min(map_bic[map_bic > 0.])
    ma_value = np.min(map_aic[map_aic > 0.])

    map_max = map_sum.iloc[np.where(map_bic == mb_value)[0]]
    map_std = map_sum.std()

    map_w_max = map_sum_w.iloc[np.where(map_bic == mb_value)[0]]
    map_w_std = map_sum_w.std()

    map_ms = {'bic': mb_value, 'aic': ma_value}

    out = {'weights': map_w_max, 'parameters': map_max, 'metrics': map_ms}
    logging.info('\nLIST OF MEDIAN MAP VARIABLES for the {} param distribution'.format(distType))

    for var, val in map_max.iteritems():
        logging.info('MAP {}: {}+-{} (max +- std)'.format(var, val.values, map_std[var]))

    logging.info('WEIGHTS MOMENTS')
    for var, val in map_w_max.iteritems():
        logging.info('MAP {}: {}+-{} (max +- std)'.format(var, val.values, map_w_std[var]))

    logging.info('BIC: {} , AIC: {} '.format(map_ms['bic'], map_ms['aic']))

    return out


# -------------- RUN MH
def run_mh(model, map_inference,
           sampler_param={'n_samples': 150000, 'burn_in': 50000, 'thin': 100,
                          'remove_outliers': False},
           distType='LogNormal'):

    logging.info('-----------------MH SAMPLER IN ACTION')

    mcmc = pm.MCMC(model)
    logging.info('Starting values of the MCMC - MAPS')
    for i, var in enumerate(model.stochastics):
        var.value = map_inference['parameters'][str(var)].values[0]

    mcmc.sample(sampler_param['n_samples'], sampler_param['burn_in'], sampler_param['thin'])

    mean_deviance = np.mean(mcmc.db.trace('deviance')(), axis=0)  # to calculating dic
    dic = 2 * mean_deviance

    mcmc.stats()
    mcmc_sum = {}
    for i, var in enumerate(model.stochastics):
        mcmc_sum[str(var)] = mcmc.trace(str(var))[:]

    mcmc_sum = pd.DataFrame.from_dict(mcmc_sum)
    mcmc_median = mcmc_sum.median()
    mcmc_std = mcmc_sum.std()

    if sampler_param['remove_outliers']:
        mcmc_sum = remove_outliers_f(mcmc_sum)

    if distType == 'TruncNormal':
        mu_sum, sigma_sum = mcf.par2mom(mcmc_sum['pw0'].values, mcmc_sum['pw1'].values, distType=distType)
        mcmc_sum_w = pd.DataFrame(np.array([mu_sum, sigma_sum]).T, columns=['A', 'stdA'])
        mcmc_sum_w['freq'] = mcmc_sum['freq']
    elif distType == 'Exponential':
        mu_sum = np.zeros(np.shape(mcmc_sum['pw0'].values))
        sigma_sum = np.zeros(np.shape(mcmc_sum['pw0'].values))
        for i, val in enumerate(mcmc_sum['pw0'].values):
            mu_sum[i], sigma_sum[i] = mcf.par2mom(mcmc_sum['pw0'].values[i], mcmc_sum['pw1'].values[i],
                                                  distType=distType)

        mcmc_sum_w = pd.DataFrame(np.array([mu_sum, sigma_sum]).T, columns=['A', 'stdA'])
        mcmc_sum_w['freq'] = mcmc_sum['freq']
    elif distType == 'LogNormal':
        mcmc_sum_w = mcmc_sum[['pw0', 'pw1', 'freq']]
        mcmc_sum_w.columns = ['A', 'stdA', 'freq']
    else:
        raise Exception('distType {} not valid'.format(distType))

    mcmc_w_median = mcmc_sum_w.median()
    mcmc_w_std = mcmc_sum_w.std()


    logging.info('\n SUMMARY OF MH MC - PARAMETERS - {} WEIGHTS DISTRIBUTION'.format(distType))
    for var, val in mcmc_median.iteritems():
        logging.info('MH {}: {}+-{} (median +- std)'.format(var, val, mcmc_std[var]))

    logging.info('\n SUMMARY OF THE USEFUL INFO OF THE INFERENCE')
    for var, val in mcmc_w_median.iteritems():
        logging.info('MH {}: {}+-{} (median +- std)'.format(var, val, mcmc_w_std[var]))

    logging.info('DIC: {}'.format(dic))

    out = {'parameters': mcmc_sum, 'weights': mcmc_sum_w, 'dic': dic, 'mcmc_object': mcmc}

    return out


# ---------------- OTHER FUNCTIONS

def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n


def exp_1(x, A, tau, c):
    A = np.abs(A)
    tau = np.abs(tau)
    c = np.abs(c)
    return A * np.exp(-x / tau) + c


def f_like(x, t1_fix, t2_fix, distType, y_target, n_moments=4):
    A = x[0]
    stdA = x[1]
    nu = x[2]

    if A < 0 or stdA < 0:
        return 10 ** 20

    if distType == 'LogNormal':
        weightType = 'Moments'
    elif distType == 'TruncNormal':
        weightType = 'Parameters'
    elif distType == 'Exponential':
        weightType = 'Parameters'

    y_theo = mcf.inputs2momenta_full(nu, A, stdA, t2_fix, t1_fix, dist_type=distType, weightType=weightType)
    return np.sum((y_theo[:n_moments] - y_target[:n_moments]) ** 2)


def remove_outliers_f(mcmc_sum):
    mcmc_quant_1 = mcmc_sum.quantile(0.02)
    mcmc_quant_2 = mcmc_sum.quantile(0.98)

    for ii in mcmc_sum.__iter__():  # I mask the top 2% and bottom 2% data
        mcmc_sum[ii] = mcmc_sum[ii].mask(mcmc_sum[ii] < mcmc_quant_1[ii])
        mcmc_sum[ii] = mcmc_sum[ii].mask(mcmc_sum[ii] > mcmc_quant_2[ii])

    return mcmc_sum.dropna()


def lognorm_param(mean, std):
    return np.log((mean ** 2) / np.sqrt(std ** 2 + mean ** 2)), np.sqrt(np.log(std ** 2 / mean ** 2 + 1));


def f_tau_min(x, freq, sigma, y_target):
    constant = x[0]
    tau2 = x[1]
    tau1 = x[2]

    if constant < 0.:
        return 10 ** 20
    if tau2 < 0.:
        return 10 ** 20
    if tau1 < 0.000:
        return 10 ** 20

    y_theo = mcf.power_spectrum_tau(freq, constant, tau2, tau1)

    return np.sum((y_theo - y_target) ** 2 / sigma ** 2)
