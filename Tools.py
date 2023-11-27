import pickle
import os
import six
import json
import numpy as np

def load_pickle_BeRNN(file):
    try:
        with open(file, 'rb') as f:
            data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', file, ':', e)
        raise
    return data

rules_dict = \
    {'BeRNN' :  ['DM', 'DM Anti', 'EF', 'EF Anti', 'RP', 'RP Anti', 'RP Ctx1',
                         'RP Ctx2', 'WM', 'WM Anti', 'WM Ctx1', 'WM Ctx2']}

def get_num_ring_BeRNN(ruleset):
    '''get number of stimulus rings'''
    return 2

def get_num_rule_BeRNN(ruleset):
    '''get number of rules'''
    return len(rules_dict[ruleset])

def get_default_hp_BeRNN(ruleset):
    '''Get a default hp.

    Useful for debugging.

    Returns:
        hp : a dictionary containing training hpuration
    '''
    num_ring = get_num_ring_BeRNN(ruleset)
    n_rule = get_num_rule_BeRNN(ruleset)

    n_eachring = 32
    n_input, n_output = 1+num_ring*n_eachring+n_rule, n_eachring+1
    hp = {
            # input type: normal, multi
            'in_type': 'normal',
            # Type of RNNs: LeakyRNN, LeakyGRU, EILeakyGRU, GRU, LSTM
            'rnn_type': 'LeakyRNN',
            # whether rule and stimulus inputs are represented separately
            'use_separate_input': False,
            # Type of loss functions
            'loss_type': 'lsq',
            # Optimizer
            'optimizer': 'adam',
            # Type of activation functions, relu, softplus, tanh, elu
            'activation': 'relu',
            # Time constant (ms)
            'tau': 100,
            # discretization time step (ms)
            'dt': 20,
            # discretization time step/time constant
            'alpha': 0.2,
            # recurrent noise
            'sigma_rec': 0.05,
            # input noise
            'sigma_x': 0.01,
            # leaky_rec weight initialization, diag, randortho, randgauss
            'w_rec_init': 'randortho',
            # a default weak regularization prevents instability
            'l1_h': 0,
            # l2 regularization on activity
            'l2_h': 0,
            # l1 regularization on weight
            'l1_weight': 0,
            # l2 regularization on weight
            'l2_weight': 0,
            # l2 regularization on deviation from initialization
            'l2_weight_init': 0,
            # proportion of weights to train, None or float between (0, 1)
            'p_weight_train': None,
            # number of units each ring
            'n_eachring': n_eachring,
            # number of rings
            'num_ring': num_ring,
            # number of rules
            'n_rule': n_rule,
            # first input index for rule units
            'rule_start': 1+num_ring*n_eachring,
            # number of input units
            'n_input': n_input,
            # number of output units
            'n_output': n_output,
            # number of recurrent units
            'n_rnn': 256,
            # random number used for several random initializations
            'rng' : np.random.RandomState(seed=0),
            # number of input units
            'ruleset': ruleset,
            # name to save
            'save_name': 'test_model',
            # learning rate
            'learning_rate': 0.001,
            # # intelligent synapses parameters, tuple (c, ksi)
            # 'c_intsyn': 0,
            # 'ksi_intsyn': 0,
            }

    return hp

def save_hp_BeRNN(hp, model_dir):
    """Save the hyper-parameter file of model save_name"""
    hp_copy = hp.copy()
    hp_copy.pop('rng')  # rng can not be serialized
    with open(os.path.join(model_dir, 'hp.json'), 'w') as f:
        json.dump(hp_copy, f)

# def mkdir_p_BeRNN(path):
#     """
#     Portable mkdir -p
#
#     """
#     try:
#         os.makedirs(path)
#     # except OSError as e:
#     #     if e.errno == errno.EEXIST and os.path.isdir(path):
#     #         pass
#     #     else:
#     #         raise

def gen_feed_dict_BeRNN(model, Input, Output, hp):
    """Generate feed_dict for session run."""
    if hp['in_type'] == 'normal':
        feed_dict = {model.x: Input,
                     model.y: Output}
    else:
        raise ValueError()

    return feed_dict

def popvec_BeRNN(y):
    """Population vector read out.

    Assuming the last dimension is the dimension to be collapsed

    Args:
        y: population output on a ring network. Numpy array (Batch, Units)

    Returns:
        Readout locations: Numpy array (Batch,)
    """
    pref = np.arange(0, 2*np.pi, 2*np.pi/y.shape[-1])  # preferences
    temp_sum = y.sum(axis=-1)
    temp_cos = np.sum(y*np.cos(pref), axis=-1)/temp_sum
    temp_sin = np.sum(y*np.sin(pref), axis=-1)/temp_sum
    loc = np.arctan2(temp_sin, temp_cos)
    return np.mod(loc, 2*np.pi)

def get_perf_BeRNN(y_hat, y_loc):
    """Get performance.

    Args:
      y_hat: Actual output. Numpy array (Time, Batch, Unit)
      y_loc: Target output location (-1 for fixation).
        Numpy array (Time, Batch)

    Returns:
      perf: Numpy array (Batch,)
    """
    if len(y_hat.shape) != 3:
        raise ValueError('y_hat must have shape (Time, Batch, Unit)')
    # Only look at last time points
    y_loc = y_loc[-1]
    y_hat = y_hat[-1]

    # Fixation and location of y_hat
    y_hat_fix = y_hat[..., 0]
    y_hat_loc = popvec_BeRNN(y_hat[..., 1:])

    # Fixating? Correctly saccading?
    fixating = y_hat_fix > 0.5

    original_dist = y_loc - y_hat_loc
    dist = np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))
    corr_loc = dist < 0.2*np.pi

    # Should fixate?
    should_fix = y_loc < 0

    # performance
    perf = should_fix * fixating + (1-should_fix) * corr_loc * (1-fixating)
    return perf

def save_log_BeRNN(log):
    """Save the log file of model."""
    model_dir = log['model_dir']
    fname = os.path.join(model_dir, 'log.json')
    with open(fname, 'w') as f:
        json.dump(log, f)

def load_log_BeRNN(model_dir):
    """Load the log file of model save_name"""
    fname = os.path.join(model_dir, 'log.json')
    if not os.path.isfile(fname):
        return None

    with open(fname, 'r') as f:
        log = json.load(f)
    return log

def load_hp_BeRNN(model_dir):
    """Load the hyper-parameter file of model save_name"""
    fname = os.path.join(model_dir, 'hp.json')
    if not os.path.isfile(fname):
        fname = os.path.join(model_dir, 'hparams.json')  # backward compat
        if not os.path.isfile(fname):
            return None

    with open(fname, 'r') as f:
        hp = json.load(f)

    # Use a different seed aftering loading,
    # since loading is typically for analysis
    hp['rng'] = np.random.RandomState(hp['seed']+1000)
    return hp
