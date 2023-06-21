import time
from collections import defaultdict

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import random
import os
# import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
# print(tf.__version__)

# import task
from network import Model
# from analysis import variance
import tools
import task
from train import get_default_hp #, do_eval_BeRNN

from Preprocessing import fileDict, prepare_DM, prepare_EF, prepare_RP, prepare_WM


# Create adapted training function
def train(model_dir, hp=None, display_step = 50, ruleset='BeRNN', rule_trains=None, rule_prob_map=None, seed=0, load_dir=None, trainables=None):
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

    # tools.mkdir_p(model_dir) # todo: create directory if not existing

    # Network parameters
    default_hp = get_default_hp(ruleset)
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    hp['seed'] = seed
    hp['rng'] = np.random.RandomState(seed)

    # Rules to train and test. Rules in a set are trained together
    if rule_trains is None:
        # By default, training all rules available to this ruleset
        hp['rule_trains'] = task.rules_dict[ruleset]
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
    tools.save_hp(hp, model_dir)

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
    random_AllTasks_list = random.sample(AllTasks_list, 100)

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
            print(step, display_step)
            print(step % display_step)
            try:
                # # Validation
                # if step % display_step == 0:
                #     log['trials'].append(step)
                #     log['times'].append(time.time() - t_start)
                #     log = do_eval_BeRNN(sess, model, log, hp['rule_trains'], AllTasks_list)
                #     # if log['perf_avg'][-1] > model.hp['target_perf']:
                #     # check if minimum performance is above target
                #     if log['perf_min'][-1] > model.hp['target_perf']:
                #         print('Perf reached the target: {:0.2f}'.format(
                #             hp['target_perf']))
                #         break

                # Training
                if currentBatch.split('_')[1] == 'DM':
                    Input, Output = prepare_DM(currentBatch, 0, 48) # co: cmask problem: (model, hp['loss_type'], currentBatch, 0, 48)
                elif currentBatch.split('_')[1] == 'EF':
                    Input, Output = prepare_EF(currentBatch, 0, 48)
                elif currentBatch.split('_')[1] == 'RP':
                    Input, Output = prepare_RP(currentBatch, 0, 48)
                elif currentBatch.split('_')[1] == 'WM':
                    Input, Output = prepare_WM(currentBatch, 0, 48)
                # Generating feed_dict.
                feed_dict = tools.gen_feed_dict_BeRNN(model, Input, Output, hp) # co: cmask problem: (model, Input, Output, c_mask, hp)
                sess.run(model.train_step, feed_dict=feed_dict)

            except BaseException as e:
                print('error with: ' + currentBatch)
                print('error message: ' + str(e))

        # Saving the model
        model.save()
        print("Optimization finished!")


# Train
model_dir_BeRNN = os.getcwd() + '\\generalModel_BeRNN\\'

train(model_dir=model_dir_BeRNN, seed=0, display_step=50, rule_trains=None, \
      rule_prob_map=None, load_dir=None,trainables=None)



# LAB ##################################################################################################################

model = Model(model_dir_BeRNN)
hp = model.hp
# validate model with do_eval_BeRNN




with tf.Session() as sess:
    model.restore()