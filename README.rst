 invivoinfer is a python package containing the code described in `Extraction of Synaptic Input Properties in-vivo (2016) P.Puggioni, M. Jelitai, I. Duguid, M. van Rossum`

** You should feel that the code is undertested and certainly sub-optimal **


Installation
------------

-  Clone this repo and run (it will install all the dependencies too)

::

    git clone https://github.com/ppuggioni/invivoinfer.git
    cd invivoinfer
    pip install -r requirements.txt
    python setup.py develop


Getting Started
---------------

- if you want, run the tests in the test folder, to make sure all is installed correctly

- open jupyter notebook

::

    jupyter notebook

and open the notebook notebooks/Example_analysis.ipynb

if you run the notebook, you should get all the plots and understand how to use the package.
Note that running the notebook as it is might be long (~15/20 minutes?). For testing purposes, you should
run with config_testing.json when initialising the class (at some point in the notebook I wrote a warning).


Important note on the config.json
---------------------------------

This is the main file where you control the options of the inference.
Probably the most important one at the beginning is the one to baseline the trace:

::

    "baseline_corr": {
    "average": 95,
    "uncertainty": 5,
    "ToUse": true}

Where `average` is the baseline and `uncertainty` is, as you expect, the uncertainty of your baseline estimation.
