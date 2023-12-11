import os
import sys
# import json
import time
# import errno
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf

from network import Model
from collections import defaultdict
import Tools
from Preprocessing_error import prepare_DM_error, prepare_EF_error, prepare_RP_error, prepare_WM_error, fileDict_error
# from Preprocessing_correct import prepare_WM_correct, prepare_DM_correct, prepare_EF_correct, prepare_RP_correct, fileDict_correct


########################################################################################################################
'''TOOLS'''
########################################################################################################################
rules_dict = \
    {'BeRNN' :  ['DM', 'DM Anti', 'EF', 'EF Anti', 'RP', 'RP Anti', 'RP Ctx1',
                         'RP Ctx2', 'WM', 'WM Anti', 'WM Ctx1', 'WM Ctx2']}


########################################################################################################################
'''Network validation'''
########################################################################################################################
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

            # todo: ####################################################################################################
            # todo: ####################################################################################################
            # co: It might be better to randomly draw the trials used for training and validation, as their might be more error in the last trials
            if currentBatch.split('_')[2] == 'DM':
                Input, Output, y_loc = prepare_DM_error(currentBatch, 48, 60)  # co: cmask problem: (model, hp['loss_type'], currentBatch, 0, 48)
            elif currentBatch.split('_')[2] == 'EF':
                Input, Output, y_loc = prepare_EF_error(currentBatch, 48, 60)
            elif currentBatch.split('_')[2] == 'RP':
                Input, Output, y_loc = prepare_RP_error(currentBatch, 48, 60)
            elif currentBatch.split('_')[2] == 'WM':
                Input, Output, y_loc = prepare_WM_error(currentBatch, 48, 60)

            feed_dict = Tools.gen_feed_dict_BeRNN(model, Input, Output, hp)
            c_lsq, c_reg, y_hat_test = sess.run(
                [model.cost_lsq, model.cost_reg, model.y_hat],
                feed_dict=feed_dict)

            # print('c_lsq, c_reg, y_hat_test = ', c_lsq, c_reg, y_hat_test)

            # Cost is first summed over time,
            # and averaged across batch and units
            # We did the averaging over time through c_mask
            perf_test = np.mean(Tools.get_perf_BeRNN(y_hat_test, y_loc))
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
    Tools.save_log_BeRNN(log)

    return log


########################################################################################################################
'''Network training'''
########################################################################################################################
# co: ALL TASKS ########################################################################################################
def train_BeRNN(model_dir, hp=None, display_step=None, ruleset='BeRNN', rule_trains=None, rule_prob_map=None, seed=0, load_dir=None, trainables=None):
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
    default_hp = Tools.get_default_hp_BeRNN(ruleset)
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
    Tools.save_hp_BeRNN(hp, model_dir)

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
    # todo: ############################################################################################################
    # todo: ############################################################################################################
    # todo: Create taskList to generate trials from
    # xlsxFolder = os.getcwd() + '\\Data CSP\\MH\\'
    xlsxFolder = 'W:\\group_csp\\analyses\\oliver.frank\\BeRNN_main\\01_1\\'
    AllTasks_list = fileDict_error(xlsxFolder)
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
        for i in range(200):
            # Validation
            log['trials'].append(i * 48 * 9)  # We have 108 .xlsx files and every time all of them are processed there are 48 trials * 9 files for each task fed to the network
            log['times'].append(time.time() - t_start)
            log = do_eval_BeRNN(sess, model, log, hp['rule_trains'], AllTasks_list)
            print('TRAINING ##########################################################################')

            # loop through all existing data
            for step in range(len(random_AllTasks_list)): # * hp['batch_size_train'] <= max_steps:
                currentBatch = random_AllTasks_list[step]
                # currentBatch = random_AllTasks_list[1]
                try:
                    # Count batches
                    batchNumber += 1
                    print('Batch #', batchNumber)
                    # Training
                    # co: It might be better to randomly draw the trials used for training and validation, as their might be more error in the last trials
                    if currentBatch.split('_')[2] == 'DM':
                        Input, Output, y_loc, epochs = prepare_DM_error(currentBatch, 0, 48) # co: cmask problem: (model, hp['loss_type'], currentBatch, 0, 48)
                    elif currentBatch.split('_')[2] == 'EF':
                        Input, Output, y_loc, epochs = prepare_EF_error(currentBatch, 0, 48)
                    elif currentBatch.split('_')[2] == 'RP':
                        Input, Output, y_loc, epochs = prepare_RP_error(currentBatch, 0, 48)
                    elif currentBatch.split('_')[2] == 'WM':
                        Input, Output, y_loc, epochs = prepare_WM_error(currentBatch, 0, 48)
                    # Generating feed_dict.
                    feed_dict = Tools.gen_feed_dict_BeRNN(model, Input, Output, hp) # co: cmask problem: (model, Input, Output, c_mask, hp)
                    sess.run(model.train_step, feed_dict=feed_dict)

                except BaseException as e:
                    print('error with: ' + currentBatch)
                    print('error message: ' + str(e))

        # Saving the model
        model.save()
        print("Optimization finished!")

# # Apply the network training
model_dir_BeRNN = os.getcwd() + '\\BeRNN_models\\01_1_1\\'
train_BeRNN(model_dir=model_dir_BeRNN, seed=0, display_step=None, rule_trains=None, rule_prob_map=None, load_dir=None, trainables=None)


