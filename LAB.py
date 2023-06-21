# BeRNN_env
########################################################################################################################
# todo: Interactive mode for matplotlib will be activated which enables scientific computing (code batch execution)
# todo: tensorflow Future warning for possible conflicts between numpy and tensorflow - silence warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import train
# Ignore the os recommandation to use the AVX2 CPU processors, as you have GPU which will run faster and is set up automatically
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
########################################################################################################################
# Create several batches for similiar tasks you also used in your experiment
########################################################################################################################
import task

# Set up default hyperparameters
default_hp = train.get_default_hp(ruleset='all')
os.getcwd()

# todo: Executive Function task: Network has to response in direction of central stim (mod1:form; mod2:strength(ignore))
# ReactGo task: Either modality 1 or 2; Network has to response in direction of presented stimulus; is enabled to directly answer after stim onset
trial_reactgo_random = task.generate_trials(rule = 'reactgo', hp = default_hp, batch_size = 64, mode = 'random', noise_on = False)
df_trial_reactgo_random_input = trial_reactgo_random.x
df_trial_reactgo_random_output = trial_reactgo_random.y
# todo: Executive Function Anti task: Network has to response in opposite direction of central stim (mod1:form; mod2:strength(ignore))
# ReactGoAnti task: Either modality 1 or 2; Network has to response in opposite direction of presented stimulus; is enabled to directly answer after stim onset
trial_reactanti_random = task.generate_trials(rule = 'reactanti', hp = default_hp, batch_size = 64, mode = 'random', noise_on = False)
df_trial_reactanti_random_input = trial_reactanti_random.x
df_trial_reactanti_random_output = trial_reactanti_random.y


# todo: Decision Making task: Network has to response in direction in which most arrows indicate (mod1:form; mod2:strength(ignore))
# todo: Decision Making Anti task: Network has to response in opposite direction in which most arrows indicate (mod1:form; mod2:strength(ignore))
# todo: Relational Processing Ctx 1 task: Network has to response in direction on one of the mostly presented forms (mod1:form; mod2:color(ignore))
# todo: Relational Processing Ctx 2 task: Network has to response in direction on one of the mostly presented colors (mod1:form(ignore); mod2:color)
# contextdm1/contextdm2 task: Ignore either modality 1 or 2; Network has to response in direction of presented stimulus with highest strength
trial_contextdm1_random = task.generate_trials(rule = 'contextdm1', hp = default_hp, batch_size = 64, mode = 'random', noise_on = False)
trial_contextdm2_random = task.generate_trials(rule = 'contextdm2', hp = default_hp, batch_size = 64, mode = 'random', noise_on = False)

# todo: Relational Processing task: Network has to response in direction of only unique presented stimulus (mod1:form; mod2:color)
# todo: Relational Processing Anti task: Network has to response in direction of one of the mostly presented stimuli (mod1:form; mod2:color)
# multidm task: Both modalities taken into concern; Network has to response in direction of presented stimulus with highest strength
trial_multidm_random = task.generate_trials(rule = 'multidm', hp = default_hp, batch_size = 64, mode = 'random', noise_on = False)

# todo: Working Memory task: Network has to response in direction of stimulus that was already shown in the stim_poch before (mod1:form; mod2:color)
# todo: Working Memory Anti task: Network has to response in direction of stimulus that was not shown in the stim_poch before (mod1:form; mod2:color)
# todo: Working Memory Ctx1 task: Network has to indicate Match(right) or Mismatch(left) if composition of colors was the same as two trials before (mod1:form(ignore); mod2:color)
# todo: Working Memory Ctx2 task: Network has to indicate Match(right) or Mismatch(left) if composition of forms was the same as two trials before (mod1:form; mod2:color(ignore))
# Matching task (DMS): Network should response in direction of second shown stim if it matches with previous one, maintain fixation otherwise
trial_dmsgo_random = task.generate_trials(rule = 'dmsgo', hp = default_hp, batch_size = 64, mode = 'random', noise_on = False)
########################################################################################################################
# todo: So every batch has the same number of time steps, with the same number of stim_on, stim_off and stim_dur steps
# todo: It is not recommandable to use different amounts of total step in one batch, for feasibility
# todo: The number of total time steps in every batch can change if the hp/config variables are set or created before loop with generate_trials()
# todo: If they are inside the loop, the numbers won't vary. The pre_response time between tasks vary, whereas the post_response time stays constant
# todo: To hold the costs for learning between tasks equally between tasks (so that one is not over-weighted while another one is under-weighted) and
# todo: give every task teh possibility to equally effect the networks topological structure we scale the pre_response times with the c_mask()
########################################################################################################################
#
#
#
# ########################################################################################################################
# # todo: Different representations of ndarry for export and BeRNN imitation step
# ########################################################################################################################
# # todo: La solucion
# # Store representation as .pkl
import numpy as np
# # Store
# with open('output.txt', 'w') as outfile:
#     # I'm writing a header here just for the sake of readability
#     # Any line starting with "#" will be ignored by numpy.loadtxt
#     outfile.write('# Array shape: {0}\n'.format(df_trial_reactgo_random_output.shape))
#
#     # Iterating through a ndimensional array produces slices along
#     # the last axis. This is equivalent to data[i,:,:] in this case
#     for data_slice in df_trial_reactgo_random_output:
#         # The formatting string indicates that I'm writing out
#         # the values in left-justified columns 7 characters in width
#         # with 2 decimal places.
#         np.savetxt(outfile, data_slice, fmt='%-7.2f')
#
#         # Writing out a break to indicate different slices...
#         outfile.write('# New slice\n')
# # # Restore
new_df_trial_reactgo_random_input = np.loadtxt('input.txt') # todo: That is a 2d array, the sequences are just vertically concatenated
old_df_trial_reactgo_random_input = new_df_trial_reactgo_random_input.reshape((104, 64, 85))
# # new_df_trial_reactgo_random_output = np.loadtxt('output.txt') # todo: That is a 2d array, the sequences are just vertically concatenated
# # old_df_trial_reactgo_random_output = new_df_trial_reactgo_random_output.reshape((104, 64, 33))
# ########################################################################################################################
#
#
# csp_frank_oliver env
########################################################################################################################
# Create similiar data structure from the .xlsx files
########################################################################################################################



# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
# # Save it
# with open('newInput.txt', 'w') as outfile:
#     # I'm writing a header here just for the sake of readability
#     # Any line starting with "#" will be ignored by numpy.loadtxt
#     outfile.write('# Array shape: {0}\n'.format(newInput.shape))
#
#     # Iterating through a ndimensional array produces slices along
#     # the last axis. This is equivalent to data[i,:,:] in this case
#     for data_slice in newInput:
#         # The formatting string indicates that I'm writing out
#         # the values in left-justified columns 7 characters in width
#         # with 2 decimal places.
#         np.savetxt(outfile, data_slice, fmt='%-7.2f')
#
#         # Writing out a break to indicate different slices...
#         outfile.write('# New slice\n')
#
# with open('newOutput.txt', 'w') as outfile:
#     # I'm writing a header here just for the sake of readability
#     # Any line starting with "#" will be ignored by numpy.loadtxt
#     outfile.write('# Array shape: {0}\n'.format(newOutput.shape))
#
#     # Iterating through a ndimensional array produces slices along
#     # the last axis. This is equivalent to data[i,:,:] in this case
#     for data_slice in newOutput:
#         # The formatting string indicates that I'm writing out
#         # the values in left-justified columns 7 characters in width
#         # with 2 decimal places.
#         np.savetxt(outfile, data_slice, fmt='%-7.2f')
#
#         # Writing out a break to indicate different slices...
#         outfile.write('# New slice\n')

# Restore current progress #############################################################################################
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# What we want
new_df_trial_reactgo_random_input = np.loadtxt('input.txt')
wanted_reactgo_random_input = new_df_trial_reactgo_random_input.reshape((104, 64, 85))
new_df_trial_reactgo_random_output = np.loadtxt('output.txt')
wanted_reactgo_random_output = new_df_trial_reactgo_random_output.reshape((104, 64, 33))
# What we have
new_df_newInput = np.loadtxt('NewInput.txt')
current_newInput = new_df_newInput.reshape((53, 60, 77))
new_df_newOutput = np.loadtxt('newOutput.txt')
current_newOutput = new_df_newOutput.reshape((53, 60, 33))




########################################################################################################################
# LAB ##################################################################################################################
########################################################################################################################

########################################################################################################################
# Calculate results of all tasks
namesList = ['JW', 'KR', 'MH', 'PG', 'SC']
# fileList = os.listdir('C:/Users/olive/OneDrive/Desktop/Pape/Data CSP/JW')
fileList = os.listdir('C:/Users/Oliver.Frank/Desktop/Data CSP/JW')
xlsxListLength = len(fileList)
# Allocate two lists for iterating through the right files and columns, respectively
fileBatches = [[0,9],[9,18],[18,27],[27,36],[36,45],[45,54],[54,63],[63,72],[72,81],[81,90],[90,99],[99,108]]
fileRow = [3,1,0,2,3,5,7,1,3,5,7,1]

for t in range(0,len(fileBatches)):
    # adjust range of j according to fileList entries
    for j in range(fileBatches[t][0],fileBatches[t][1]): # .xlsx files that should be opened
        performance = 0
        performanceList = []
        for i in namesList:
            # folder = 'C:/Users/olive/OneDrive/Desktop/Pape/Data CSP/' + i
            folder = 'C:/Users/Oliver.Frank/Desktop/Data CSP/' + i
            filesList = os.listdir(folder)
            print(filesList[j])
            file = folder + '/' + filesList[j]
            cFile = pd.read_excel(file, engine='openpyxl')
            # Get individual performance on all trials
            percentCorrect_cols = [col for col in cFile if 'Store: PercentCorrect' in col]
            currentPerformance = cFile[percentCorrect_cols[fileRow[t]]].tail(2).iloc[0] # row that is taken into account
            print(percentCorrect_cols[fileRow[t]])
            performance += currentPerformance
            performanceList.append(currentPerformance)
        # Average performance over all participants
        finalPerformance = performance/ len(namesList)
        stdPerformance = np.std(performanceList)
        print('########## ' + filesList[j] + ' final performance:')
        print(finalPerformance)
        print(stdPerformance) # todo: Or better plot the variance?

    # Create variance plot for every





########################################################################################################################
# GRAVEYARD ############################################################################################################
########################################################################################################################
# train.train(model_dir=model_dir, hp={'learning_rate': 0.001}, ruleset='mante')
# train.train_sequential(model_dir=model_dir, hp={'learning_rate': 0.001}, ruleset='mante')
#
# import tools
# import pandas as pd
#
# # Look at the log file
# log = tools.load_log(model_dir) # The log file gives us the progress of training the networks
#
# df_log = pd.DataFrame(
#     {'trials': log['trials'],
#      'times': log['times'],
#      'cost_contextdm1': log['cost_contextdm1'],
#      'creg_contextdm1': log['creg_contextdm1'],
#      'perf_contextdm1': log['perf_contextdm1'],
#      'cost_contextdm2': log['cost_contextdm2'],
#      'creg_contextdm2': log['creg_contextdm2'],
#      'perf_contextdm2': log['perf_contextdm2'],
#      'perf_avg': log['perf_avg'],
#      'perf_min': log['perf_min']
#     })
########################################################################################################################
# # import ..
# # # run simulation
# # setting = train.get_default_hp('mante')
# model_dir = 'C:/Users/Oliver.Frank/PycharmProjects/multitask_old-master/train_all/test_model'
# hp_dir = model_dir + '/hp.json'
# # data = contextdm_data_analysis.run_simulation(save_name = model_dir, setting = setting)
#
# import json
# # Open json file
# os.chdir(model_dir)
# f = open('hp.json')
# hp = json.load(f)
# # # Iterating through the json
# # for i in hp['n_input']:
# #  print(i)
# # Closing file
# f.close()
########################################################################################################################
# GRAVEYARD ############################################################################################################
########################################################################################################################


def train(model_dir,
          hp=None,
          max_steps=1e7,
          display_step=500,
          ruleset='mante',
          rule_trains=None,
          rule_prob_map=None,
          seed=0,
          rich_output=False,
          load_dir=None,
          trainables=None,
          ):
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

    tools.mkdir_p(model_dir)

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

    # Assign probabilities for rule_trains. # todo: Here you can add probabilities e.g. to fix over representation of certain tasks
    if rule_prob_map is None:
        rule_prob_map = dict()

    # Turn into rule_trains format
    hp['rule_probs'] = None
    if hasattr(hp['rule_trains'], '__iter__'):
        # Set default as 1.
        rule_prob = np.array(
            [rule_prob_map.get(r, 1.) for r in hp['rule_trains']])
        hp['rule_probs'] = list(rule_prob / np.sum(rule_prob))
    tools.save_hp(hp, model_dir)

    # Build the model
    model = Model(model_dir, hp=hp)

    # Display hp
    for key, val in hp.items():
        print('{:20s} = '.format(key) + str(val))

    # Store results
    log = defaultdict(list)
    log['model_dir'] = model_dir

    # Record time
    t_start = time.time()

    # todo: Create taskList to generate trials from
    xlsxFolder = 'C:/Users/olive/OneDrive/Desktop/Pape/Data CSP/'  # HOME
    xlsxFolderList = os.listdir(xlsxFolder)
    random_AllTasks_list = fileDict(xlsxFolder, xlsxFolderList)

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

        step = 0
        while step * hp['batch_size_train'] <= max_steps:
            try:
                # Validation
                if step % display_step == 0:
                    log['trials'].append(step * hp['batch_size_train'])
                    log['times'].append(time.time() - t_start)
                    log = do_eval(sess, model, log, hp['rule_trains'])
                    # if log['perf_avg'][-1] > model.hp['target_perf']:
                    # check if minimum performance is above target
                    if log['perf_min'][-1] > model.hp['target_perf']:
                        print('Perf reached the target: {:0.2f}'.format(
                            hp['target_perf']))
                        break

                    if rich_output:
                        display_rich_output(model, sess, step, log, model_dir)

                # Training
                rule_train_now = hp['rng'].choice(hp['rule_trains'],
                                                  p=hp['rule_probs'])
                # Generate a random batch of trials.
                currentBatch = random.choice(random_AllTasks_list,1)

                if currentBatch.split('_')[1] == 'DM':
                    currentBatch = random_AllTasks_list
                    Input, Output = prepare_DM(currentBatch, 48, 60)
                elif currentBatch.split('_')[1] == 'EF':
                    Input, Output = prepare_EF(currentBatch, 48, 60)
                elif currentBatch.split('_')[1] == 'RP':
                    Input, Output = prepare_RP(currentBatch, 48, 60)
                elif currentBatch.split('_')[1] == 'WM':
                    Input, Output = prepare_WM(currentBatch, 48, 60)
                # Generating feed_dict.
                feed_dict = tools.gen_feed_dict_BeRNN(model, Input, Output, hp)
                sess.run(model.train_step, feed_dict=feed_dict)

                step += 1

            except KeyboardInterrupt:
                print("Optimization interrupted by user")
                break

        print("Optimization finished!")







"""Add a cost mask.

        Usually there are two periods, pre and post response
        Scale the mask weight for the post period so in total it's as important
        as the pre period
        """

        pre_on   = int(100/20) # never check the first 100ms
        pre_offs = [70] * 48
        post_ons = [30] * 48
        # var = [var] * self.batch_size
        c_mask = np.zeros((100, 48, 33))
        for i in range(48):
            # Post response periods usually have the same length across tasks
            c_mask[post_ons[i]:, i, :] = 5.
            # Pre-response periods usually have different lengths across tasks
            # todo: Reason for c_mask: To keep cost comparable across tasks
            # Scale the cost mask of the pre-response period by a factor
            c_mask[pre_on:pre_offs[i], i, :] = 1.

        # self.c_mask[:, :, 0] *= self.n_eachring # Fixation is important
        c_mask[:, :, 0] *= 2. # Fixation is important

        c_mask = c_mask.reshape((100*48, 33))




def do_eval(sess, model, log, rule_train):
    """Do evaluation.

    Args:
        sess: tensorflow session
        model: Model class instance
        log: dictionary that stores the log
        rule_train: string or list of strings, the rules being trained
    """
    hp = model.hp
    if not hasattr(rule_train, '__iter__'):
        rule_name_print = rule_train
    else:
        rule_name_print = ' & '.join(rule_train)

    print('Trial {:7d}'.format(log['trials'][-1]) +
          '  | Time {:0.2f} s'.format(log['times'][-1]) +
          '  | Now training '+rule_name_print)

    for rule_test in hp['rules']:
        n_rep = 16
        batch_size_test_rep = int(hp['batch_size_test']/n_rep)
        clsq_tmp = list()
        creg_tmp = list()
        perf_tmp = list()
        for i_rep in range(n_rep):
            trial = generate_trials(
                rule_test, hp, 'random', batch_size=batch_size_test_rep)
            feed_dict = tools.gen_feed_dict(model, trial, hp)
            c_lsq, c_reg, y_hat_test = sess.run(
                [model.cost_lsq, model.cost_reg, model.y_hat],
                feed_dict=feed_dict)

            # Cost is first summed over time,
            # and averaged across batch and units
            # We did the averaging over time through c_mask
            perf_test = np.mean(get_perf(y_hat_test, trial.y_loc))
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

    # TODO: This needs to be fixed since now rules are strings
    if hasattr(rule_train, '__iter__'):
        rule_tmp = rule_train
    else:
        rule_tmp = [rule_train]
    perf_tests_mean = np.mean([log['perf_'+r][-1] for r in rule_tmp])
    log['perf_avg'].append(perf_tests_mean)

    perf_tests_min = np.min([log['perf_'+r][-1] for r in rule_tmp])
    log['perf_min'].append(perf_tests_min)

    # Saving the model
    model.save()
    tools.save_log(log)

    return log













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



































