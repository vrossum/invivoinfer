from invivoinfer.vc_infer_trace import VCTrace
import imp
import os
import numpy as np
import logging
import json
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

CONFIG_DIR = os.path.join("/".join(imp.find_module("invivoinfer")[1].split('/')[:-1]), 'config')
DATA_DIR = os.path.join('/'.join(imp.find_module("invivoinfer")[1].split('/')[:-1]), 'data')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_trace.csv')

def test_raw_trace():
    trace = np.genfromtxt(TEST_DATA_PATH, delimiter=',')
    dt = 0.0001
    infertrace = VCTrace(trace, dt=dt)
    assert infertrace.dt == 0.0001
    assert len(infertrace.trace) == len(trace)

    config_file = os.path.join(CONFIG_DIR, 'config.json')
    with open(config_file, 'r') as fp:
        out = json.load(fp)

    assert out == infertrace.default_params