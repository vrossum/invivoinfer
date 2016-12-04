from __future__ import division

import numpy as np
np.seterr(divide = 'warn')
from invivoinfer.infer import VCInfer


class VCAnalysis(VCInfer):
    """
    Voltage clamp raw_trace
    """
    def __init__(self, trace, dt, n_moments=4, params=None, figures_folder='figure_output', config=None):
        """
        :param trace: np.array of the raw_trace
        :param dt: time step in seconds
        """
        super(VCAnalysis, self).__init__(trace, dt, params=params, figures_folder=figures_folder, config=config)


    def run_analysis(self, save_pdf=False, distType='LogNormal', sampler='mh'):

        self.momenta()
        self.psd_calc()
        self.estimate_taus()

        self.reset_obs_unc()

        # ADD BASELINE
        self.set_obs_unc_mean()

        # CORRECT FOR LF AND HF NOISE
        self.set_obs_unc_std()

        # FIRST GUESS  (ML fit)
        self.first_guess(distType=distType)

        # OPTIMIZE LIKELIHOOD
        self.optimize_likelihood()

        # CREATE MODEL AND SAMPLE
        self.create_model()

        self.run_sampler(sampler_type=sampler)







