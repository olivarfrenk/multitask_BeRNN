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
from Preprocessing import fileDict, prepare_DM, prepare_EF, prepare_RP, prepare_WM


# todo: Create taskList to generate trials from
xlsxFolder = os.getcwd() + '\\Data CSP\\'
xlsxFolderList = os.listdir(os.getcwd() + '\\Data CSP\\')
AllTasks_list = fileDict(xlsxFolder, xlsxFolderList)
random_AllTasks_list = random.sample(AllTasks_list, len(AllTasks_list))

for step in range(len(random_AllTasks_list)):  # * hp['batch_size_train'] <= max_steps:
    currentBatch = random_AllTasks_list[step]
    print('Batch #', step)
    try:
        # Validation
        if step % display_step == 0:
            log['trials'].append(step)
            log['times'].append(time.time() - t_start)
            log = do_eval_BeRNN(sess, model, log, hp['rule_trains'], AllTasks_list)
            print('TRAINING ##########################################################################')

        # Training
        if currentBatch.split('_')[2] == 'DM':
            Input, Output, y_loc = prepare_DM(currentBatch, 0,48)  # co: cmask problem: (model, hp['loss_type'], currentBatch, 0, 48)
        elif currentBatch.split('_')[2] == 'EF':
            Input, Output, y_loc = prepare_EF(currentBatch, 0, 48)
        elif currentBatch.split('_')[2] == 'RP':
            Input, Output, y_loc = prepare_RP(currentBatch, 0, 48)
        elif currentBatch.split('_')[2] == 'WM':
            Input, Output, y_loc = prepare_WM(currentBatch, 0, 48)
        # Generating feed_dict.
        feed_dict = gen_feed_dict_BeRNN(model, Input, Output, hp)  # co: cmask problem: (model, Input, Output, c_mask, hp)
        sess.run(model.train_step, feed_dict=feed_dict)
