import imp
import os
import numpy as np
import logging
import json

from invivoinfer.infer import VCInfer, _plot_mc_results

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

CONFIG_DIR = os.path.join("/".join(imp.find_module("invivoinfer")[1].split('/')[:-1]), 'config')
DATA_DIR = os.path.join('/'.join(imp.find_module("invivoinfer")[1].split('/')[:-1]), 'data')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_trace.csv')


def test_init():
    raw_trace = np.genfromtxt(TEST_DATA_PATH, delimiter=',')
    dt = 0.0001

    vcinfer = VCInfer(raw_trace, dt)

    assert vcinfer.dt == dt
    assert vcinfer.n_moments == 4

def test_momenta():
    raw_trace = np.genfromtxt(TEST_DATA_PATH, delimiter=',')
    dt = 0.0001

    vcinfer = VCInfer(raw_trace, dt)
    vcinfer.momenta()

    keys = ['mean', 'std', 'skew', 'kurtosis']
    for key in keys:
        assert key in vcinfer.descr_stat


def test_psd():
    raw_trace = np.genfromtxt(TEST_DATA_PATH, delimiter=',')
    dt = 0.0001

    vcinfer = VCInfer(raw_trace, dt)
    vcinfer.psd_calc()

    keys = ['psd_x', 'psd_x_freq', 'psd_x_std']
    for key in keys:
        assert key in vcinfer.descr_ps

def test_estimate_tau():
    raw_trace = np.genfromtxt(TEST_DATA_PATH, delimiter=',')
    dt = 0.0001

    vcinfer = VCInfer(raw_trace, dt)
    vcinfer.psd_calc()
    vcinfer.estimate_taus()
    vcinfer.plot_tau_estimation(save_pdf=True)
    assert vcinfer.param_priors['tau1'] > 0.
    assert vcinfer.param_priors['tau2'] > 0.
    assert vcinfer.param_priors['constant'] > 0.

def test_set_obs_unc_mean():

    config_file = os.path.join(CONFIG_DIR, 'config.json')

    with open(config_file, mode='r') as fp:
        config = json.load(fp)

    raw_trace = np.genfromtxt(TEST_DATA_PATH, delimiter=',')
    dt = 0.0001

    vcinfer = VCInfer(raw_trace, dt)
    vcinfer.reset_obs_unc()
    vcinfer.set_obs_unc_mean()
    vcinfer.plot_mean_offset(save_pdf=True)
    assert vcinfer.obs_bias['mean'][0] == config['baseline_corr']['average']
    assert vcinfer.obs_bias['std'][0] == config['baseline_corr']['uncertainty']


def test_set_obs_unc_std():

    raw_trace = np.genfromtxt(TEST_DATA_PATH, delimiter=',')
    dt = 0.0001

    vcinfer = VCInfer(raw_trace, dt)
    vcinfer.momenta()
    vcinfer.psd_calc()
    vcinfer.estimate_taus()
    vcinfer.reset_obs_unc()
    assert vcinfer.obs_bias['mean'][1] ==0
    vcinfer.set_obs_unc_std()
    assert vcinfer.obs_bias['mean'][1] > 0

def test_first_guess():

    raw_trace = np.genfromtxt(TEST_DATA_PATH, delimiter=',')
    dt = 0.0001

    vcinfer = VCInfer(raw_trace, dt)
    vcinfer.momenta()
    vcinfer.psd_calc()
    vcinfer.estimate_taus()
    vcinfer.reset_obs_unc()
    vcinfer.set_obs_unc_mean()
    vcinfer.set_obs_unc_std()

    dist_list = ['LogNormal', 'Exponential', 'TruncNormal']

    for dist_type in dist_list:
        logging.info('******************** Inference using {} weights'.format(dist_type))
        vcinfer.first_guess(distType=dist_type)
        assert vcinfer.param_priors['pw0'][0] > 0.
        assert vcinfer.param_priors['pw1'][0] > 0.
        assert vcinfer.param_priors['freq'][0] > 0.


def test_likelihood_est_f():
    config_file = os.path.join(CONFIG_DIR, 'config.json')

    with open(config_file, mode='r') as fp:
        config = json.load(fp)


    raw_trace = np.genfromtxt(TEST_DATA_PATH, delimiter=',')
    dt = 0.0001

    vcinfer = VCInfer(raw_trace, dt)
    vcinfer.momenta()
    vcinfer.psd_calc()
    vcinfer.estimate_taus()
    vcinfer.reset_obs_unc()
    vcinfer.set_obs_unc_mean()
    vcinfer.set_obs_unc_std()

    vcinfer.first_guess(distType='LogNormal')
    vcinfer.optimize_likelihood()
    vcinfer.plot_likelihood(save_pdf=True)

    assert len(vcinfer.likelihood_samples.index) == config['likelihood_par']['nsamples']
    assert 'kde_ll' in vcinfer.likelihood_estim
    assert 'moments_sim' in vcinfer.likelihood_estim

    assert len(vcinfer.likelihood_estim['moments_sim']['mean']) == vcinfer.n_moments


def test_model():
    raw_trace = np.genfromtxt(TEST_DATA_PATH, delimiter=',')
    dt = 0.0001

    vcinfer = VCInfer(raw_trace, dt, config='config_testing.json')
    vcinfer.momenta()
    vcinfer.psd_calc()
    vcinfer.estimate_taus()
    vcinfer.reset_obs_unc()
    vcinfer.set_obs_unc_mean()
    vcinfer.set_obs_unc_std()

    dist_list = ['LogNormal', 'TruncNormal']#, 'Exponential' it takes ages

    for dist_type in dist_list:
        vcinfer.first_guess(distType=dist_type)
        vcinfer.optimize_likelihood()

        vcinfer.create_model()
        vcinfer.run_sampler()
        vcinfer.plot_mc_results(save_pdf=True)
        assert 'map' in vcinfer.inference_results
        assert 'mh' in vcinfer.inference_results

        for el in ['weights', 'parameters', 'metrics']:
            assert el in vcinfer.inference_results['map']

        for el in ['weights', 'parameters', 'dic', 'mcmc_object']:
            assert el in vcinfer.inference_results['mh']


def test_plot_mh():
    import pickle
    with open(os.path.join(DATA_DIR,'map_par.pkl'), 'r') as fp:
        map_data = pickle.load(fp)
    with open(os.path.join(DATA_DIR, 'mh_par.pkl'), 'r') as fp:
        mh_data = pickle.load(fp)

    _plot_mc_results(map_data, mh_data, save_pdf=True, figure_dir=os.path.join(DATA_DIR, 'figure_output'))
