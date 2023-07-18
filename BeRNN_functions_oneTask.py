import os
import sys
import json
import time
# import errno
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf

from network import Model
from collections import defaultdict
from Preprocessing import fileDict, prepare_DM, prepare_EF, prepare_RP, prepare_WM
# Interactive mode for matplotlib will be activated which enables scientific computing (code batch execution)
import matplotlib.pyplot as plt


########################################################################################################################
'''TOOLS'''
########################################################################################################################
rule_dict = \
    {'BeRNN' :  ['WM']}

def get_num_ring_BeRNN(ruleset):
    '''get number of stimulus rings'''
    return 2

def get_num_rule_BeRNN(ruleset):
    '''get number of rules'''
    return len(rule_dict[ruleset])

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

    print('VALIDATION ##########################################################################')
    print('Trial {:7d}'.format(log['trials'][-1] * 60) +      # [-1] calls the last element of a list
          '  | Time {:0.2f} s'.format(log['times'][-1]) +
          '  | Now training ' + rule_name_print)

    for rule_test in hp['rules']:

        n_rep = 5 # how often 12 trials from every task are drawn for model validation

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

    return log


########################################################################################################################
'''Network training'''
########################################################################################################################

# co: ONE TASK #########################################################################################################
def train_BeRNN_oneTask(model_dir, hp=None, display_step = 5, ruleset='BeRNN', rule_trains=None, rule_prob_map=None, seed=0, load_dir=None, trainables=None):
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

    # mkdir_p_BeRNN(model_dir) # todo: create directory if not existing

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
        hp['rule_trains'] = rule_dict[ruleset]
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
    xlsxFolder = os.getcwd() + '/Data CSP/'
    xlsxFolderList = os.listdir(os.getcwd() + '/Data CSP/')
    AllTasks_list = fileDict(xlsxFolder, xlsxFolderList)
    random_AllTasks_list = random.sample(AllTasks_list, len(AllTasks_list))

    WM_list = []
    WM_Anti_list = []
    WM_Ctx1_list = []
    WM_Ctx2_list = []

    for i in random_AllTasks_list:
        print(i.split('_')[2], i.split('_')[3])
        if i.split('_')[2] == 'WM' and i.split('_')[3] != 'Anti' and i.split('_')[3] != 'Ctx1' and i.split('_')[3] != 'Ctx2':
            WM_list.append(i)
        elif i.split('_')[2] == 'WM' and i.split('_')[3] == 'Anti':
            WM_Anti_list.append(i)
        elif i.split('_')[2] == 'WM' and i.split('_')[3] == 'Ctx1':
            WM_Ctx1_list.append(i)
        elif i.split('_')[2] == 'WM' and i.split('_')[3] == 'Ctx2':
            WM_Ctx2_list.append(i)

    with tf.Session() as sess:
        if load_dir is not None:
            model.restore(load_dir)  # complete restore
        else:
            # Assume everything is restored
            sess.run(tf.global_variables_initializer())

        # Set trainable parameters
        if trainables is None or trainables == 'all':
            var_list = model.var_list  # train everything
        else:
            raise ValueError('Unknown trainables')
        # actualizes the network optimization method and variable list (only important if that changes)
        model.set_optimizer(var_list=var_list)

        # penalty on deviation from initial weight
        # iteratively reduces weight strength so that the model complexity is reduced - prevents overfitting
        if hp['l2_weight_init'] > 0:
            anchor_ws = sess.run(model.weight_list)
            for w, w_val in zip(model.weight_list, anchor_ws):
                model.cost_reg += (hp['l2_weight_init'] *
                                   tf.nn.l2_loss(w - w_val))

            model.set_optimizer(var_list=var_list)

        # partial weight training
        # laying weight mask with L2 regularization term over all trainable weights
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

        batchNumber = 0
        # loop through all existing data several times
        for i in range(50):
            # loop through all existing data
            for step in range(len(WM_list)): # * hp['batch_size_train'] <= max_steps:
                currentBatch = WM_list[step]
                try:
                    # Validation
                    if step % display_step == 0:
                        log['trials'].append(batchNumber*48)   # Average trials per batch fed to network on one task (48/12)
                        log['times'].append(time.time() - t_start)
                        log = do_eval_BeRNN(sess, model, log, hp['rule_trains'], WM_list)
                        print('TRAINING ##########################################################################')

                    # Count batches
                    batchNumber += 1
                    print('Batch #', batchNumber)
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
model_dir_BeRNN = os.getcwd() + '/BeRNN_models/generalModel_CSP_20_WM_pan/'
train_BeRNN_oneTask(model_dir=model_dir_BeRNN, seed=0, display_step=5, rule_trains=None, rule_prob_map=None, load_dir=None, trainables=None)

########################################################################################################################
'''Network analysis'''
########################################################################################################################
# Analysis functions
_rule_color = {
            'DM': 'green',
            'DM Anti': 'olive',
            'EF': 'forest green',
            'EF Anti': 'mustard',
            'RP': 'tan',
            'RP Anti': 'brown',
            'RP Ctx1': 'lavender',
            'RP Ctx2': 'aqua',
            'WM': 'bright purple',
            'WM Anti': 'green blue',
            'WM Ctx1': 'blue',
            'WM Ctx2': 'indigo'
            }

rule_color = {k: 'xkcd:'+v for k, v in _rule_color.items()}

def easy_activity_plot_BeRNN(model_dir, rule):
    """A simple plot of neural activity from one task.

    Args:
        model_dir: directory where model file is saved
        rule: string, the rule to plot
    """

    model = Model(model_dir)
    hp = model.hp

    xlsxFolder = os.getcwd() + '\\Data CSP\\'
    xlsxFolderList = os.listdir(os.getcwd() + '\\Data CSP\\')
    AllTasks_list = fileDict(xlsxFolder, xlsxFolderList)
    random_AllTasks_list = random.sample(AllTasks_list, len(AllTasks_list))

    with tf.Session() as sess:
        model.restore()

        currentRule = ' '
        while currentRule != rule:
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
        h, y_hat = sess.run([model.h, model.y_hat], feed_dict=feed_dict)
        # All matrices have shape (n_time, n_condition, n_neuron)

    # Take only the one example trial
    i_trial = 10

    for activity, title in zip([Input, h, y_hat],
                               ['input', 'recurrent', 'output']):
        plt.figure()
        plt.imshow(activity[:,i_trial,:].T, aspect='auto', cmap='hot',      # np.uint8
                   interpolation='none', origin='lower')
        plt.title(title)
        plt.colorbar()
        plt.show()

def plot_performanceprogress_BeRNN(model_dir, rule_plot=None):
    # Plot Training Progress
    log = load_log_BeRNN(model_dir)
    hp = load_hp_BeRNN(model_dir)

    trials = log['trials']  #

    fs = 6 # fontsize
    fig = plt.figure(figsize=(3.5,1.2))
    ax = fig.add_axes([0.1,0.25,0.35,0.6])
    lines = list()
    labels = list()

    x_plot = np.array(trials)
    if rule_plot == None:
        rule_plot = hp['rules']

    for i, rule in enumerate(rule_plot):
        # line = ax1.plot(x_plot, np.log10(cost_tests[rule]),color=color_rules[i%26])
        # ax2.plot(x_plot, perf_tests[rule],color=color_rules[i%26])
        line = ax.plot(x_plot, np.log10(log['cost_'+rule]),
                       color=rule_color[rule])
        ax.plot(x_plot, log['perf_'+rule], color=rule_color[rule])
        lines.append(line[0])
        labels.append(rule)

    ax.tick_params(axis='both', which='major', labelsize=fs)

    ax.set_ylim([0, 1])
    ax.set_xlabel('Trials per task',fontsize=fs, labelpad=2)
    ax.set_ylabel('Performance',fontsize=fs, labelpad=0)
    ax.locator_params(axis='x', nbins=3)
    ax.set_yticks([0,1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    lg = fig.legend(lines, labels, title='Task',ncol=2,bbox_to_anchor=(0.47,0.5),
                    fontsize=fs,labelspacing=0.3,loc=6,frameon=False)
    plt.setp(lg.get_title(),fontsize=fs)
    # Add the randomness thresholds
    # DM & RP Ctx
    plt.axhline(y=0.2, color='green', label= 'DM & DM Anti & RP Ctx1 & RP Ctx2', linestyle=':')
    # EF
    plt.axhline(y=0.25, color='black', label= 'EF & EF Anti', linestyle=':')
    # RP
    plt.axhline(y=0.143, color='brown', label= 'RP & RP Anti', linestyle=':')
    # WM
    plt.axhline(y=0.5, color='blue', label= 'WM & WM Anti & WM Ctx1 & WM Ctx2', linestyle=':')

    rt = fig.legend(title='Randomness threshold', bbox_to_anchor=(0.47, 0.4), fontsize=fs, labelspacing=0.3, loc=6, frameon=False)
    plt.setp(rt.get_title(), fontsize=fs)

    plt.show()


model_dir = os.getcwd() + '/BeRNN_models/generalModel_CSP_20_WM_pan'
rule = 'DM'
# Plot activity of input, recurrent and output layer for one test trial
easy_activity_plot_BeRNN(model_dir, rule)
# Plot improvement of performance over iterating training steps
plot_performanceprogress_BeRNN(model_dir)


