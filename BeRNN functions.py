import os
import sys
import json
import time
import errno
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf

from network import Model
from collections import defaultdict
from Preprocessing import fileDict, prepare_DM, prepare_EF, prepare_RP, prepare_WM


########################################################################################################################
'''TOOLS'''
########################################################################################################################
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
            # todo: added by Oliver - is used to randomly choose the position of the stimuli on the circle in the trials
            'rng' : np.random.RandomState(seed=0),
            # number of input units
            'ruleset': ruleset,
            # name to save
            'save_name': 'test_model',
            # learning rate
            'learning_rate': 0.001,
            # intelligent synapses parameters, tuple (c, ksi)
            'c_intsyn': 0,
            'ksi_intsyn': 0,
            }

    return hp

def save_hp_BeRNN(hp, model_dir):
    """Save the hyper-parameter file of model save_name"""
    hp_copy = hp.copy()
    hp_copy.pop('rng')  # rng can not be serialized
    with open(os.path.join(model_dir, 'hp.json'), 'w') as f:
        json.dump(hp_copy, f)

def mkdir_p_BeRNN(path):
    """
    Portable mkdir -p

    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def gen_feed_dict_BeRNN(model, Input, Output, hp):
    """Generate feed_dict for session run."""
    if hp['in_type'] == 'normal':
        feed_dict = {model.x: Input,
                     model.y: Output}
    elif hp['in_type'] == 'multi':
        n_time, batch_size = Input.shape[:2]
        new_shape = [n_time,
                     batch_size,
                     hp['rule_start']*hp['n_rule']]

        x = np.zeros(new_shape, dtype=np.float32)
        for i in range(batch_size):
            ind_rule = np.argmax(Input[0, i, hp['rule_start']:])
            i_start = ind_rule*hp['rule_start']
            x[:, i, i_start:i_start+hp['rule_start']] = \
                Input[:, i, :hp['rule_start']]

        feed_dict = {model.x: x,
                     model.y: Output}
    else:
        raise ValueError()

    return feed_dict


########################################################################################################################
'''Network validation'''
########################################################################################################################
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
    y_hat_loc = popvec_BeRNN(y_hat[..., 1:])    # debugging evaluation: popvec_BeRNN(y_hat_test[..., 1:])

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

def do_eval_BeRNN(sess, model, log, rule_train, AllTasks_list):
    """Do evaluation.

    Args:
        sess: tensorflow session
        model: Model class instance
        log: dictionary that stores the log
        rule_train: string or list of strings, the rules being trained
    """

    # For debugging ####################################################################################################
    # log = defaultdict(list)
    # xlsxFolder = os.getcwd() + '\\Data CSP\\'
    # xlsxFolderList = os.listdir(os.getcwd() + '\\Data CSP\\')
    # AllTasks_list = fileDict(xlsxFolder, xlsxFolderList)
    # # load pre-trained model
    # # BeRNN
    # model_dir_BeRNN = os.getcwd() + '\\generalModel_BeRNN\\'
    # model = Model(model_dir_BeRNN)
    # hp = model.hp
    # hp['rule_trains'] = rules_dict['BeRNN']
    # rule_train = hp['rule_trains']
    # with tf.Session() as sess:
    #     model.restore(model_dir_BeRNN)
    # # Yang

    ###################################################################################################################
    hp = model.hp

    if not hasattr(rule_train, '__iter__'):
        rule_name_print = rule_train
    else:
        rule_name_print = ' & '.join(rule_train)

    print('VALIDATION')
    print('Trial {:7d}'.format(log['trials'][-1]) +      # [-1] calls the last element of a list
          '  | Time {:0.2f} s'.format(log['times'][-1]) +
          '  | Now training ' + rule_name_print)

    for rule_test in hp['rules']:
        print(rule_test)

        n_rep = 1 # how often 12 trials from every task are drawn for model validation

        clsq_tmp = list()
        creg_tmp = list()
        perf_tmp = list()

        # Get right trials from files for testing
        for i_rep in range(n_rep):
            currentRule = ' '
            while currentRule != rule_test:
                currentBatch = random.sample(AllTasks_list, 1)[0]
                if len(currentBatch.split('_')) == 6:
                    currentRule = currentBatch.split('_')[2] + ' ' + currentBatch.split('_')[3]
                else:
                    currentRule = currentBatch.split('_')[2]

            if currentBatch.split('_')[2] == 'DM':
                Input, Output, y_loc = prepare_DM(currentBatch, 48, 60)  # co: cmask problem: (model, hp['loss_type'], currentBatch, 0, 48)
            elif currentBatch.split('_')[2] == 'EF':
                Input, Output, y_loc = prepare_EF(currentBatch, 48, 60)
            elif currentBatch.split('_')[2] == 'RP':
                Input, Output, y_loc = prepare_RP(currentBatch, 48, 60)
            elif currentBatch.split('_')[2] == 'WM':
                Input, Output, y_loc = prepare_WM(currentBatch, 48, 60)

            feed_dict = gen_feed_dict_BeRNN(model, Input, Output, hp)
            c_lsq, c_reg, y_hat_test = sess.run(
                [model.cost_lsq, model.cost_reg, model.y_hat],
                feed_dict=feed_dict)

            # print('c_lsq, c_reg, y_hat_test = ', c_lsq, c_reg, y_hat_test)

            # Cost is first summed over time,
            # and averaged across batch and units
            # We did the averaging over time through c_mask
            perf_test = np.mean(get_perf_BeRNN(y_hat_test, y_loc))
            clsq_tmp.append(c_lsq)
            creg_tmp.append(c_reg)
            perf_tmp.append(perf_test)

        log['cost_'+rule_test].append(np.mean(clsq_tmp, dtype=np.float64))
        log['creg_'+rule_test].append(np.mean(creg_tmp, dtype=np.float64))
        log['perf_'+rule_test].append(np.mean(perf_tmp, dtype=np.float64))
        print('{:15s}'.format(rule_test) +
              '| cost {:0.6f}'.format(np.mean(clsq_tmp)) +
              '| c_reg {:0.6f}'.format(np.mean(creg_tmp)) +
              '  | perf {:0.2f}'.format(np.mean(perf_tmp)))
        print('Log: ', log)
        sys.stdout.flush()


        # # TODO: This needs to be fixed since now rules are strings
        # if hasattr(rule_train, '__iter__'):
        #     rule_tmp = rule_train
        # else:
        #     rule_tmp = [rule_train]
        # perf_tests_mean = np.mean([log['perf_'+r][-1] for r in rule_tmp])
        # log['perf_avg'].append(perf_tests_mean)
        #
        # perf_tests_min = np.min([log['perf_'+r][-1] for r in rule_tmp])
        # log['perf_min'].append(perf_tests_min)

        # Saving the model
        model.save()
        save_log_BeRNN(log)

        # return log


########################################################################################################################
'''Network training'''
########################################################################################################################
def train_BeRNN(model_dir, hp=None, display_step = 5, ruleset='BeRNN', rule_trains=None, rule_prob_map=None, seed=0, load_dir=None, trainables=None):
    """Train the network.

    Args:
        model_dir: str, training directory
        hp: dictionary of hyperparameters
        max_steps: int, maximum number of training steps
        display_step: int, display steps
        ruleset: the set of rules to train
        rule_trains: list of rules to train, if None then all rules possible
        rule_prob_map: None or dictionary of relative rule probability
        seed: int, random seed to be used

    Returns:
        model is stored at model_dir/model.ckpt
        training configuration is stored at model_dir/hp.json
    """

    mkdir_p_BeRNN(model_dir) # todo: create directory if not existing

    # Network parameters
    default_hp = get_default_hp_BeRNN(ruleset)
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    hp['seed'] = seed
    hp['rng'] = np.random.RandomState(seed)

    # Rules to train and test. Rules in a set are trained together
    if rule_trains is None:
        # By default, training all rules available to this ruleset
        hp['rule_trains'] = rules_dict[ruleset]
    else:
        hp['rule_trains'] = rule_trains
    hp['rules'] = hp['rule_trains']

    # Assign probabilities for rule_trains.
    # todo: Here you can add probabilities e.g. to fix over representation of certain tasks
    # todo: by default it is evenly distributed
    if rule_prob_map is None:
        rule_prob_map = dict()

    # Turn into rule_trains format
    hp['rule_probs'] = None
    if hasattr(hp['rule_trains'], '__iter__'):
        # Set default as 1.
        rule_prob = np.array([rule_prob_map.get(r, 1.) for r in hp['rule_trains']])
        hp['rule_probs'] = list(rule_prob / np.sum(rule_prob))
    save_hp_BeRNN(hp, model_dir)

    # Build the model
    model = Model(model_dir, hp=hp)         # todo: include self.cmask

    # Display hp
    for key, val in hp.items():
        print('{:20s} = '.format(key) + str(val))

    # Store results
    log = defaultdict(list)
    log['model_dir'] = model_dir

    # Record time
    t_start = time.time()

    # todo: Create taskList to generate trials from
    xlsxFolder = os.getcwd() + '\\Data CSP\\'
    xlsxFolderList = os.listdir(os.getcwd() + '\\Data CSP\\')
    AllTasks_list = fileDict(xlsxFolder, xlsxFolderList)
    random_AllTasks_list = random.sample(AllTasks_list, len(AllTasks_list))

    with tf.Session() as sess:
        if load_dir is not None:
            model.restore(load_dir)  # complete restore
        else:
            # Assume everything is restored
            sess.run(tf.global_variables_initializer())

        # Set trainable parameters
        if trainables is None or trainables == 'all':
            var_list = model.var_list  # train everything
        elif trainables == 'input':
            # train all nputs
            var_list = [v for v in model.var_list
                        if ('input' in v.name) and ('rnn' not in v.name)]
        elif trainables == 'rule':
            # train rule inputs only
            var_list = [v for v in model.var_list if 'rule_input' in v.name]
        else:
            raise ValueError('Unknown trainables')
        model.set_optimizer(var_list=var_list)

        # penalty on deviation from initial weight
        if hp['l2_weight_init'] > 0:
            anchor_ws = sess.run(model.weight_list)
            for w, w_val in zip(model.weight_list, anchor_ws):
                model.cost_reg += (hp['l2_weight_init'] *
                                   tf.nn.l2_loss(w - w_val))

            model.set_optimizer(var_list=var_list)

        # partial weight training
        if ('p_weight_train' in hp and
                (hp['p_weight_train'] is not None) and
                hp['p_weight_train'] < 1.0):
            for w in model.weight_list:
                w_val = sess.run(w)
                w_size = sess.run(tf.size(w))
                w_mask_tmp = np.linspace(0, 1, w_size)
                hp['rng'].shuffle(w_mask_tmp)
                ind_fix = w_mask_tmp > hp['p_weight_train']
                w_mask = np.zeros(w_size, dtype=np.float32)
                w_mask[ind_fix] = 1e-1  # will be squared in l2_loss
                w_mask = tf.constant(w_mask)
                w_mask = tf.reshape(w_mask, w.shape)
                model.cost_reg += tf.nn.l2_loss((w - w_val) * w_mask)
            model.set_optimizer(var_list=var_list)


        for step in range(len(random_AllTasks_list)): # * hp['batch_size_train'] <= max_steps:
            currentBatch = random_AllTasks_list[step]
            print('Batch #',step)
            print('Log: ',log)
            try:
                # Validation
                if step % display_step == 0:
                    log['trials'].append(step)
                    log['times'].append(time.time() - t_start)
                    log = do_eval_BeRNN(sess, model, log, hp['rule_trains'], AllTasks_list)
                    print('Log after evaluation: ', log)

                # Training
                if currentBatch.split('_')[2] == 'DM':
                    Input, Output, y_loc = prepare_DM(currentBatch, 0, 48) # co: cmask problem: (model, hp['loss_type'], currentBatch, 0, 48)
                elif currentBatch.split('_')[2] == 'EF':
                    Input, Output, y_loc = prepare_EF(currentBatch, 0, 48)
                elif currentBatch.split('_')[2] == 'RP':
                    Input, Output, y_loc = prepare_RP(currentBatch, 0, 48)
                elif currentBatch.split('_')[2] == 'WM':
                    Input, Output, y_loc = prepare_WM(currentBatch, 0, 48)
                # Generating feed_dict.
                feed_dict = gen_feed_dict_BeRNN(model, Input, Output, hp) # co: cmask problem: (model, Input, Output, c_mask, hp)
                sess.run(model.train_step, feed_dict=feed_dict)

            except BaseException as e:
                print('error with: ' + currentBatch)
                print('error message: ' + str(e))

        # Saving the model
        model.save()
        print("Optimization finished!")

# Apply the network training
model_dir_BeRNN = os.getcwd() + '\\generalModel_BeRNN\\'
train_BeRNN(model_dir=model_dir_BeRNN, seed=0, display_step=5, rule_trains=None, rule_prob_map=None, load_dir=None, trainables=None)


########################################################################################################################
'''LAB'''
########################################################################################################################
# model_dir_BeRNN = os.getcwd() + '\\generalModel_BeRNN\\'
# model = Model(model_dir_BeRNN)
# hp = model.hp
#
# with tf.Session() as sess:
#     model.restore()