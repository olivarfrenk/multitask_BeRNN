########################################################################################################################
# todo: Create input and output data structure on the collected data according to Yang form ############################
########################################################################################################################
import numpy as np
import pandas as pd
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


########################################################################################################################
# co: Step 1: Prepare collected data equally to Yang form ##############################################################
########################################################################################################################
# Gradient activation function
def get_dist(original_dist):
    '''Get the distance in periodic boundary conditions'''
    return np.minimum(abs(original_dist),2*np.pi-abs(original_dist))
def add_x_loc(x_loc, pref):
    """Input activity given location."""
    dist = get_dist(x_loc - pref)  # periodic boundary
    dist /= np.pi / 8
    return 0.8 * np.exp(-dist ** 2 / 2)

# Preperation functions ################################################################################################

# DM & DM Anti
def prepare_DM(file_location, sequence_on, sequence_off): # (model, loss_type, file_location, sequence_on, sequence_off)
    # For bug fixing
    file_location, sequence_on, sequence_off = os.getcwd() + '\\Data CSP\\JW\\7962306_DM_easy_1100.xlsx', 0, 48
    # Open .xlsx and select necessary columns
    df = pd.read_excel(file_location, engine='openpyxl')
    # Add all necessary columns to create the Yang form later
    df.loc[:, 'Fixation input'] = 1  # means: should fixate, no response
    df.loc[:, 'Fixation output'] = 0.8  # means: should response, no fixation
    df.loc[:, 'DM'] = 0
    df.loc[:, 'DM Anti'] = 0
    df.loc[:, 'EF'] = 0
    df.loc[:, 'EF Anti'] = 0
    df.loc[:, 'RP'] = 0
    df.loc[:, 'RP Anti'] = 0
    df.loc[:, 'RP Ctx1'] = 0
    df.loc[:, 'RP Ctx2'] = 0
    df.loc[:, 'WM'] = 0
    df.loc[:, 'WM Anti'] = 0
    df.loc[:, 'WM Ctx1'] = 0
    df.loc[:, 'WM Ctx2'] = 0
    # Reorder columns
    df_selection = df[['Spreadsheet', 'TimeLimit', 'Onset Time', 'Response', 'Spreadsheet: CorrectAnswer', 'Correct',
                       'Component Name', \
                       'Fixation input', 'Fixation output', 'Spreadsheet: Field 1', 'Spreadsheet: Field 2',
                       'Spreadsheet: Field 3', 'Spreadsheet: Field 4', \
                       'Spreadsheet: Field 5', 'Spreadsheet: Field 6', 'Spreadsheet: Field 7', 'Spreadsheet: Field 8',
                       'Spreadsheet: Field 9', \
                       'Spreadsheet: Field 10', 'Spreadsheet: Field 11', 'Spreadsheet: Field 12',
                       'Spreadsheet: Field 13', 'Spreadsheet: Field 14', \
                       'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17',
                       'Spreadsheet: Field 18', 'Spreadsheet: Field 19', \
                       'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22',
                       'Spreadsheet: Field 23', 'Spreadsheet: Field 24', \
                       'Spreadsheet: Field 25', 'Spreadsheet: Field 26', 'Spreadsheet: Field 27',
                       'Spreadsheet: Field 28', 'Spreadsheet: Field 29', \
                       'Spreadsheet: Field 30', 'Spreadsheet: Field 31', 'Spreadsheet: Field 32', \
                       'Spreadsheet: Field 1', 'Spreadsheet: Field 2', 'Spreadsheet: Field 3', 'Spreadsheet: Field 4', \
                       'Spreadsheet: Field 5', 'Spreadsheet: Field 6', 'Spreadsheet: Field 7', 'Spreadsheet: Field 8',
                       'Spreadsheet: Field 9', \
                       'Spreadsheet: Field 10', 'Spreadsheet: Field 11', 'Spreadsheet: Field 12',
                       'Spreadsheet: Field 13', 'Spreadsheet: Field 14', \
                       'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17',
                       'Spreadsheet: Field 18', 'Spreadsheet: Field 19', \
                       'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22',
                       'Spreadsheet: Field 23', \
                       'Spreadsheet: Field 24', 'Spreadsheet: Field 25', 'Spreadsheet: Field 26',
                       'Spreadsheet: Field 27', 'Spreadsheet: Field 28', \
                       'Spreadsheet: Field 29', 'Spreadsheet: Field 30', 'Spreadsheet: Field 31',
                       'Spreadsheet: Field 32', 'DM', \
                       'DM Anti', 'EF', 'EF Anti', 'RP', 'RP Anti', 'RP Ctx1', 'RP Ctx2', 'WM', 'WM Anti', 'WM Ctx1',
                       'WM Ctx1']]
    # Prepare incrementation for trials between fixation events in df
    incrementList = []
    for i in range(0, len(df_selection)):
        if df_selection['Component Name'][
            i] == 'Fixation':  # Get all rows with that string as they start a trial sequence
            incrementList.append(i + 1)

    # Get average epoch time steps for the selected task in one session
    finalTrialsList = []
    numFixStepsTotal = 0
    numRespStepsTotal = 0
    iterationSteps = 0
    for i in incrementList:
        currentTrial = df_selection[i:i + 2].reset_index().drop(columns=['index'])
        iterationSteps = iterationSteps + 1
        numFixSteps = round(currentTrial['Onset Time'][0] / 20)  # equal to neuronal time constant of 20ms (Yang, 2019)
        numFixStepsTotal = numFixStepsTotal + numFixSteps
        numRespSteps = round(currentTrial['Onset Time'][1] / 20)  # equal to neuronal time constant of 20ms (Yang, 2019)
        numRespStepsTotal = numRespStepsTotal + numRespSteps

    # numFixStepsAverage = round(numFixStepsTotal/iterationSteps)
    # numRespStepsAverage = round(numRespStepsTotal/iterationSteps)
    # TotalStepsAverage = numFixStepsAverage + numRespStepsAverage

    # todo: For bug fixing we will create equal sequence length for all sessions
    numFixStepsAverage = 30
    numRespStepsAverage = 70
    TotalStepsAverage = numFixStepsAverage + numRespStepsAverage

    # Take all trials and high-sample them to the average steps
    for i in incrementList:
        currentTrial = df_selection[i:i + 1].reset_index().drop(columns=['index'])
        # Create List with high-sampled rows for both epochs
        currentSequenceList = []
        for j in range(0, TotalStepsAverage):  # Add time steps
            sequence = [currentTrial.iloc[0]]
            currentSequenceList.append(sequence)
        # Append current trial to final list - corresponds to one batch/ one task in one session
        finalTrialsList.append(currentSequenceList)

    # Create final df for INPUT and OUPUT
    newOrderList = []
    # Append all the time steps accordingly to a list
    for j in range(0, TotalStepsAverage):
        for i in range(0, len(finalTrialsList)):
            newOrderList.append(finalTrialsList[i][j])

    # Create Yang form
    finalTrialsList_array = np.array(newOrderList).reshape((len(finalTrialsList[0]), len(finalTrialsList), 85))
    # Create one input file and one output file
    Input = finalTrialsList_array
    Output = finalTrialsList_array
    # Create final Input form
    Input = np.delete(Input, [0, 1, 2, 3, 4, 5, 6, 8], axis=2)
    # Create final output form
    Output = np.delete(Output, np.s_[0, 1, 2, 4, 5, 6, 7], axis=2)
    Output = np.delete(Output, np.s_[34:78], axis=2)
    # Delet all rows that are not needed (for either training or testing)
    Input = Input[:, sequence_on:sequence_off, :]
    Output = Output[:, sequence_on:sequence_off, :]

    # INPUT ############################################################################################################
    # float all fixation input values to 1
    for i in range(0, Input.shape[0]):
        for j in range(0, Input.shape[1]):
            Input[i][j][0] = float(1)
    # float all task values to 0
    for i in range(0, Input.shape[0]):
        for j in range(0, Input.shape[1]):
            for k in range(65, Input.shape[2]):
                Input[i][j][k] = float(0)
    # Float current task value to 1
    taskDict = {
        'DM ': 65,
        'DM Anti ': 66,
        'EF ': 67,
        'EF Anti ': 68,
        'RP ': 69,
        'RP Anti ': 70,
        'RP Ctx1 ': 71,
        'RP Ctx2 ': 72,
        'WM ': 73,
        'WM Anti ': 74,
        'WM Ctx1 ': 75,
        'WM Ctx2 ': 76
    }
    # float all task values to 0
    for i in range(0, Input.shape[0]):
        for j in range(0, Input.shape[1]):
            Input[i][j][taskDict[df['Spreadsheet'][0].split('-')[0]]] = float(1)
    # float all NaN's on field units to 0
    for i in range(0, Input.shape[0]):
        for j in range(0, Input.shape[1]):
            for k in range(1, 65):
                if Input[i][j][k] == 'NaN.png':
                    Input[i][j][k] = float(0)
    # float all values on mod1 fields to their true value (arrow direction)
    mod1Dict = {
        'R': float(0.25),
        'D': float(0.5),
        'L': float(0.75),
        'U': float(1.0),
    }
    for i in range(0, Input.shape[0]):
        for j in range(0, Input.shape[1]):
            for k in range(1, 33):
                if Input[i][j][k] != 0:
                    Input[i][j][k] = mod1Dict[Input[i][j][k].split('w')[0]]
    # float all values on mod2 fields to their true value (arrow strength)
    mod2Dict = {
        '0_25.png': float(0.25),
        '0_5.png': float(0.5),
        '0_75.png': float(0.75),
        '1_0.png': float(1.0),
    }
    for i in range(0, Input.shape[0]):
        for j in range(0, Input.shape[1]):
            for k in range(33, 65):
                if Input[i][j][k] != 0:
                    Input[i][j][k] = mod2Dict[Input[i][j][k].split('w')[1]]

    # float all field values of fixation period to 0
    for i in range(0, numFixStepsAverage):
        for j in range(0, Input.shape[1]):
            for k in range(1, 65):
                Input[i][j][k] = float(0)

    # Add input gradient activation
    # Create default hyperparameters for network
    num_ring, n_eachring, n_rule = 2, 32, 12
    n_input, n_output = 1 + num_ring * n_eachring + n_rule, n_eachring + 1
    pref = np.arange(0, 2 * np.pi, 2 * np.pi / 32)

    for i in range(0, Input.shape[0]):
        for j in range(0, Input.shape[1]):
            currentTimeStepModOne = Input[i][j][1:33]
            currentTimeStepModTwo = Input[i][j][33:65]
            # Allocate first unit ring
            unitRingMod1 = np.zeros(32, dtype='float32')
            unitRingMod2 = np.zeros(32, dtype='float32')

            # Get non-zero values of time steps on both modalities and also directly fix distance between units issue
            NonZero_Mod1 = np.nonzero(currentTimeStepModOne)[0]
            NonZero_Mod2 = np.nonzero(currentTimeStepModTwo)[0]
            if len(NonZero_Mod1) != 0:
                # Accumulating all activities for both unit rings together
                for k in range(0, len(NonZero_Mod1)):
                    currentStimLoc_Mod1 = pref[NonZero_Mod1[k]]
                    currentStimStrength_Mod1 = currentTimeStepModOne[NonZero_Mod1[k]]
                    currentStimLoc_Mod2 = pref[NonZero_Mod2[k]]
                    currentStimStrength_Mod2 = currentTimeStepModTwo[NonZero_Mod2[k]]
                    # add one gradual activated stim to final form
                    currentActivation_Mod1 = add_x_loc(currentStimLoc_Mod1, pref) * currentStimStrength_Mod1
                    currentActivation_Mod2 = add_x_loc(currentStimLoc_Mod2, pref) * currentStimStrength_Mod2
                    # Add all activations for one trial together
                    unitRingMod1 = np.around(unitRingMod1 + currentActivation_Mod1, decimals=2)
                    unitRingMod2 = np.around(unitRingMod2 + currentActivation_Mod2, decimals=2)
                # For bug fixing
                # print('after: ', NonZero_Mod1)
                # print('     after: ', NonZero_Mod1)

                # Store
                currentFinalRow = np.concatenate((Input[i][j][0:1], unitRingMod1, unitRingMod2, Input[i][j][65:78]))
                Input[i][j][0:78] = currentFinalRow

    # Change dtype of every element in matrix to float32 for later validation functions
    for i in range(0, Input.shape[0]):
        for j in range(0, Input.shape[1]):
            for k in range(0, Input.shape[2]):
                Input[i][j][k] = np.float32(Input[i][j][k])
    # Also change dtype for entire array
    Input = Input.astype('float32')

    # Sanity check
    print('Input solved: ', df['Spreadsheet'][0], ' ', df['TimeLimit'][0])

    # OUTPUT ###########################################################################################################
    # float all field units during fixation epoch on 0.05
    for i in range(0, numFixStepsAverage):
        for j in range(0, Output.shape[1]):
            for k in range(2, 34):
                Output[i][j][k] = float(0.05)
    # float all field units of response epoch to 0
    for i in range(numFixStepsAverage, TotalStepsAverage):
        for j in range(0, Output.shape[1]):
            for k in range(2, 34):
                Output[i][j][k] = float(0)
    # float all fixation outputs during response period to 0.05
    for i in range(numFixStepsAverage, TotalStepsAverage):
        for j in range(0, Output.shape[1]):
            Output[i][j][1] = float(0.05)

    # Assign field units to their according participant response value after fixation period
    outputDict = {
        'U': 32,
        'R': 8,
        'L': 24,
        'D': 16
    }

    for i in range(numFixStepsAverage, TotalStepsAverage):
        for j in range(0, Output.shape[1]):
            if Output[i][j][0] != 'NoResponse':
                Output[i][j][outputDict[Output[i][j][0]]] = float(0.85)
            else:
                for k in range(2, 34):
                    Output[i][j][k] = float(0.05)

    # Drop unnecessary first column
    Output = np.delete(Output, [0], axis=2)
    # Pre-allocate y-loc matrix; needed for later validation
    y_loc = np.zeros((Output.shape[0], Output.shape[1]))

    # Add output gradient activation
    for i in range(0, Output.shape[0]):
        for j in range(0, Output.shape[1]):
            currentTimeStepOutput = Output[i][j][1:33]
            # Allocate first unit ring
            unitRingOutput = np.zeros(32, dtype='float32')
            # Get non-zero values of time steps
            nonZerosOutput = np.nonzero(currentTimeStepOutput)[0]
            # Float first fixations rows with -1
            for k in range(0, numFixStepsAverage):
                y_loc[k][j] = float(-1)

            if len(nonZerosOutput) != 0 and currentTimeStepOutput[0] != 0.05:
                # Get activity and model gradient activation around it
                currentOutputLoc = pref[nonZerosOutput[0]]
                currentActivation_Output = add_x_loc(currentOutputLoc, pref) + 0.05  # adding noise
                unitRingOutput = np.around(unitRingOutput + currentActivation_Output, decimals=2)
                # Store
                currentFinalRow = np.concatenate((Output[i][j][0:1], unitRingOutput))
                Output[i][j][0:33] = currentFinalRow
                # Complete y_loc matrix
                for k in range(numFixStepsAverage, TotalStepsAverage):
                    y_loc[k][j] = pref[nonZerosOutput[0]]

    # Change dtype of every element in matrix to float32 for later validation functions
    for i in range(0, Output.shape[0]):
        for j in range(0, Output.shape[1]):
            for k in range(0, Output.shape[2]):
                Output[i][j][k] = np.float32(Output[i][j][k])
    # Also change dtype for entire array
    Output = Output.astype('float32')

        # Sanity check
    print('Output solved: ', df['Spreadsheet'][0], ' ', df['TimeLimit'][0])



    # Create c_mask
    # c_mask = add_c_mask_BeRNN(Output.shape[0], Output.shape[1], n_output, loss_type, numFixStepsAverage, numRespStepsAverage)

    return Input, Output, y_loc    #, c_mask

# EF & EF Anti
def prepare_EF(file_location, sequence_on, sequence_off):
    # For bug fixing
    # file_location, sequence_on, sequence_off = os.getcwd() + '\\Data CSP\\JW\\7962306_EF_normal_1100.xlsx', 0, 48
    # Open .xlsx and select necessary columns
    df = pd.read_excel(file_location, engine='openpyxl')
    # Add all necessary columns to create the Yang form later
    df.loc[:, 'Fixation input'] = 1 # means: should fixate, no response
    df.loc[:, 'Fixation output'] = 0.8 # means: should response, no fixation
    df.loc[:, 'DM'] = 0
    df.loc[:, 'DM Anti'] = 0
    df.loc[:, 'EF'] = 0
    df.loc[:, 'EF Anti'] = 0
    df.loc[:, 'RP'] = 0
    df.loc[:, 'RP Anti'] = 0
    df.loc[:, 'RP Ctx1'] = 0
    df.loc[:, 'RP Ctx2'] = 0
    df.loc[:, 'WM'] = 0
    df.loc[:, 'WM Anti'] = 0
    df.loc[:, 'WM Ctx1'] = 0
    df.loc[:, 'WM Ctx2'] = 0
    # Reorder columns
    df_selection = df[['Spreadsheet', 'TimeLimit', 'Onset Time', 'Response', 'Spreadsheet: CorrectAnswer', 'Correct', 'Component Name', \
                       'Fixation input', 'Fixation output', 'Spreadsheet: Field 1', 'Spreadsheet: Field 2', 'Spreadsheet: Field 3', 'Spreadsheet: Field 4',\
                       'Spreadsheet: Field 5', 'Spreadsheet: Field 6', 'Spreadsheet: Field 7', 'Spreadsheet: Field 8', 'Spreadsheet: Field 9',\
                       'Spreadsheet: Field 10', 'Spreadsheet: Field 11', 'Spreadsheet: Field 12', 'Spreadsheet: Field 13', 'Spreadsheet: Field 14',\
                       'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17', 'Spreadsheet: Field 18', 'Spreadsheet: Field 19',\
                       'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22', 'Spreadsheet: Field 23', 'Spreadsheet: Field 24',\
                       'Spreadsheet: Field 25', 'Spreadsheet: Field 26', 'Spreadsheet: Field 27', 'Spreadsheet: Field 28', 'Spreadsheet: Field 29',\
                       'Spreadsheet: Field 30', 'Spreadsheet: Field 31', 'Spreadsheet: Field 32', \
                       'Spreadsheet: Field 1', 'Spreadsheet: Field 2', 'Spreadsheet: Field 3', 'Spreadsheet: Field 4', \
                       'Spreadsheet: Field 5', 'Spreadsheet: Field 6', 'Spreadsheet: Field 7', 'Spreadsheet: Field 8', 'Spreadsheet: Field 9', \
                       'Spreadsheet: Field 10', 'Spreadsheet: Field 11', 'Spreadsheet: Field 12', 'Spreadsheet: Field 13', 'Spreadsheet: Field 14',\
                       'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17', 'Spreadsheet: Field 18', 'Spreadsheet: Field 19',\
                       'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22', 'Spreadsheet: Field 23',\
                       'Spreadsheet: Field 24', 'Spreadsheet: Field 25', 'Spreadsheet: Field 26', 'Spreadsheet: Field 27', 'Spreadsheet: Field 28',\
                       'Spreadsheet: Field 29', 'Spreadsheet: Field 30', 'Spreadsheet: Field 31', 'Spreadsheet: Field 32', 'DM',\
                       'DM Anti', 'EF', 'EF Anti', 'RP', 'RP Anti', 'RP Ctx1', 'RP Ctx2', 'WM', 'WM Anti', 'WM Ctx1', 'WM Ctx1']]
    # Prepare incrementation for trials between fixation events in df
    incrementList = []
    for i in range(0,len(df_selection)):
        if df_selection['Component Name'][i] == 'Fixation': # Get all rows with that string as they start a trial sequence
            incrementList.append(i)

    # Get average epoch time steps for the selected task in one session
    finalTrialsList = []
    numFixStepsTotal = 0
    numRespStepsTotal = 0
    iterationSteps = 0
    for i in incrementList:
        currentTrial = df_selection[i:i + 2].reset_index().drop(columns=['index'])
        iterationSteps = iterationSteps + 1
        numFixSteps = round(currentTrial['Onset Time'][0]/20) # equal to neuronal time constant of 20ms (Yang, 2019)
        numFixStepsTotal = numFixStepsTotal + numFixSteps
        numRespSteps = round(currentTrial['Onset Time'][1]/20) # equal to neuronal time constant of 20ms (Yang, 2019)
        numRespStepsTotal = numRespStepsTotal + numRespSteps

    # numFixStepsAverage = round(numFixStepsTotal/iterationSteps)
    # numRespStepsAverage = round(numRespStepsTotal/iterationSteps)
    # TotalStepsAverage = numFixStepsAverage + numRespStepsAverage

    numFixStepsAverage = 30
    numRespStepsAverage = 70
    TotalStepsAverage = numFixStepsAverage + numRespStepsAverage

    # Take all trials and high-sample them to the average steps
    for i in incrementList:
        currentTrial = df_selection[i+1:i+2].reset_index().drop(columns=['index'])
        # Create List with high-sampled rows for both epochs
        currentSequenceList = []
        for i in range(0, numFixStepsAverage+numRespStepsAverage):
            sequence = [currentTrial.iloc[0]]
            currentSequenceList.append(sequence)
        # Append current trial to final list - corresponds to one batch/ one task in one session
        finalTrialsList.append(currentSequenceList)


    # Create final df for INPUT and OUPUT
    newOrderList = []
    # Append all the time steps accordingly to a list
    for j in range(0, TotalStepsAverage):
        for i in range(0, len(finalTrialsList)):
            newOrderList.append(finalTrialsList[i][j])

    # Create Yang form
    finalTrialsList_array = np.array(newOrderList).reshape((len(finalTrialsList[0]),len(finalTrialsList),85))
    # Create one input file and one output file
    Input = finalTrialsList_array
    Output = finalTrialsList_array
    # Create final Input form
    Input = np.delete(Input,[0,1,2,3,4,5,6,8],axis = 2)
    # Create final output form
    Output = np.delete(Output,np.s_[0,1,2,4,5,6,7],axis = 2)
    Output = np.delete(Output,np.s_[34:78],axis = 2)
    # Delete all rows that are not needed (for either training or testing)
    Input = Input[:, sequence_on:sequence_off, :]
    Output = Output[:, sequence_on:sequence_off, :]

    # INPUT ############################################################################################################
    # float all fixation input values to 1
    for i in range(0,Input.shape[0]):
        for j in range(0,Input.shape[1]):
            Input[i][j][0] = float(1)
    # float all task values to 0
    for i in range(0,Input.shape[0]):
        for j in range(0,Input.shape[1]):
            for k in range(65,Input.shape[2]):
                Input[i][j][k] = float(0)
    # Float current task value to 1
    taskDict = {
      'DM ': 65,
      'DM Anti ': 66,
      'EF ': 67,
      'EF Anti ': 68,
      'RP ': 69,
      'RP Anti ': 70,
      'RP Ctx1 ': 71,
      'RP Ctx2 ': 72,
      'WM ': 73,
      'WM Anti ': 74,
      'WM Ctx1 ': 75,
      'WM Ctx2 ': 76
    }
    # float task values to 1
    for i in range(0,Input.shape[0]):
        for j in range(0,Input.shape[1]):
            Input[i][j][taskDict[df['Spreadsheet'][0].split('-')[0]]] = float(1)
    # float all NaN's on field units to 0
    for i in range(0,Input.shape[0]):
        for j in range(0,Input.shape[1]):
            for k in range(1,65):
                if Input[i][j][k] == 'NaN.png':
                    Input[i][j][k] = float(0)
    # float all values on mod1 fields to their true value (arrow direction)
    mod1Dict = {
      'R': float(0.2),
      'D': float(0.4),
      'L': float(0.6),
      'U': float(0.8),
      'X': float(1.0)
    }
    for i in range(0,Input.shape[0]):
        for j in range(0,Input.shape[1]):
            for k in range(1,33):
                if Input[i][j][k] != 0:
                    Input[i][j][k] = mod1Dict[Input[i][j][k].split('w')[0]]
    # float all values on mod2 fields to their true value (arrow strength)
    mod2Dict = {
      '0_25.png': float(0.2),
      '0_5.png': float(0.4),
      '0_75.png': float(0.6),
      '1_0.png': float(0.8),
      'X.png': float(1.0)
    }
    for i in range(0,Input.shape[0]):
        for j in range(0,Input.shape[1]):
            for k in range(33,65):
                if Input[i][j][k] != 0:
                    Input[i][j][k] = mod2Dict[Input[i][j][k].split('w')[1]]

    # float all field values of fixation period to 0
    for i in range(0,numFixStepsAverage):
        for j in range(0,Input.shape[1]):
            for k in range(1,65):
                    Input[i][j][k] = float(0)


    # Add input gradient activation
    # Create default hyperparameters for network
    num_ring, n_eachring, n_rule = 2, 32, 12
    n_input, n_output = 1 + num_ring * n_eachring + n_rule, n_eachring + 1
    pref = np.arange(0, 2 * np.pi, 2 * np.pi / 32)

    for i in range(0, Input.shape[0]):
        for j in range(0,Input.shape[1]):
            currentTimeStepModOne = Input[i][j][1:33]
            currentTimeStepModTwo = Input[i][j][33:65]
            # Allocate first unit ring
            unitRingMod1 = np.zeros(32, dtype='float32')
            unitRingMod2 = np.zeros(32, dtype='float32')

            # Get non-zero values of time steps on both modalities and also directly fix distance between units issue
            nonZerosBeforeModOne = np.nonzero(currentTimeStepModOne)[0]
            nonZerosBeforeModTwo = np.nonzero(currentTimeStepModTwo)[0]

            ############################################################################################################
            # According to difficulty level of .xlsx choose right arrays for distance fix
            if df['Spreadsheet'][0].split('-')[1] == ' easy':
                vec1, vec2 = [1, 3, 31], [0, 2, 30]
                fix1, fix2 = [3, 0, -3], [0, -3, 3]
            elif df['Spreadsheet'][0].split('-')[1] == ' normal':
                vec1, vec2 = [1, 3, 5, 29, 31], [0, 2, 4, 28, 30]
                fix1, fix2 = [6, 3, 0, -3, -6], [0, -3, -6, 6, 3]
            elif df['Spreadsheet'][0].split('-')[1] == ' hard':
                vec1, vec2 = [1, 3, 5, 7, 27, 29, 31], [0, 2, 4, 6, 26, 28, 30]
                fix1, fix2 = [9, 6, 3, 0, -3, -6, -9], [0, -3, -6, -9, 9, 6, 3]


            if len(np.nonzero(currentTimeStepModOne)[0]) != 0:
                if np.array_equiv(nonZerosBeforeModOne, vec1) == False and np.array_equiv(nonZerosBeforeModOne, vec2) == False:
                    NonZero_Mod1 = (nonZerosBeforeModOne - fix1) % 32
                    NonZero_Mod2 = (nonZerosBeforeModTwo - fix1) % 32
                    # print('after: ', NonZero_Mod1)
                else:
                    NonZero_Mod1 = (nonZerosBeforeModOne - fix2) % 32
                    NonZero_Mod2 = (nonZerosBeforeModTwo - fix2) % 32
                    # print('     after: ', NonZero_Mod1)
                # Accumulating all activities for both unit rings together
                for k in range(0,len(NonZero_Mod1)):
                    currentStimLoc_Mod1 = pref[NonZero_Mod1[k]]
                    currentStimStrength_Mod1 = currentTimeStepModOne[nonZerosBeforeModOne[k]]
                    currentStimLoc_Mod2 = pref[NonZero_Mod2[k]]
                    currentStimStrength_Mod2 = currentTimeStepModTwo[nonZerosBeforeModTwo[k]]
                    # add one gradual activated stim to final form
                    currentActivation_Mod1 = add_x_loc(currentStimLoc_Mod1, pref) * currentStimStrength_Mod1
                    currentActivation_Mod2 = add_x_loc(currentStimLoc_Mod2, pref) * currentStimStrength_Mod2
                    # Add all activations for one trial together
                    unitRingMod1 = np.around(unitRingMod1 + currentActivation_Mod1, decimals=2)
                    unitRingMod2 = np.around(unitRingMod2 + currentActivation_Mod2, decimals=2)
                # Store
                currentFinalRow = np.concatenate((Input[i][j][0:1], unitRingMod1, unitRingMod2, Input[i][j][65:78]))
                Input[i][j][0:78] = currentFinalRow

    # Change dtype of every element in matrix to float32 for later validation functions
    for i in range(0, Input.shape[0]):
        for j in range(0, Input.shape[1]):
            for k in range(0, Input.shape[2]):
                Input[i][j][k] = np.float32(Input[i][j][k])
    # Also change dtype for entire array
    Input = Input.astype('float32')

    # Sanity check
    print('Input solved: ', df['Spreadsheet'][0], ' ', df['TimeLimit'][0])

    # OUTPUT ###########################################################################################################
    # float all field units during fixation epoch on 0.05
    for i in range(0, numFixStepsAverage):
        for j in range(0,Output.shape[1]):
            for k in range(2, 34):
                Output[i][j][k] = float(0.05)
    # float all field units of response epoch to 0
    for i in range(numFixStepsAverage,TotalStepsAverage):
        for j in range(0,Output.shape[1]):
            for k in range(2, 34):
                Output[i][j][k] = float(0)
    # float all fixation outputs during response period to 0.05
    for i in range(numFixStepsAverage,TotalStepsAverage):
        for j in range(0,Output.shape[1]):
                Output[i][j][1] = float(0.05)

    # Assign field units to their according participant response value after fixation period
    outputDict = {
      'U': 32,
      'R': 8,
      'L': 24,
      'D': 16
    }

    for i in range(numFixStepsAverage,TotalStepsAverage):
        for j in range(0,Output.shape[1]):
            if Output[i][j][0] != 'X':
                Output[i][j][outputDict[Output[i][j][0]]] = float(0.85)
            else:
                for k in range(2,34):
                    Output[i][j][k] = float(0.05)

    # Drop unnecessary first column
    Output = np.delete(Output,[0],axis = 2)
    # Pre-allocate y-loc matrix; needed for later validation
    y_loc = np.zeros((Output.shape[0], Output.shape[1]))

    # Add output gradient activation
    for i in range(0, Output.shape[0]):
        for j in range(0,Output.shape[1]):
            currentTimeStepOutput = Output[i][j][1:33]
            # Allocate first unit ring
            unitRingOutput = np.zeros(32, dtype='float32')
            # Get non-zero values of time steps
            nonZerosOutput = np.nonzero(currentTimeStepOutput)[0]

            # Float first fixations rows with -1 for validation matrix y-loc
            for k in range(0, numFixStepsAverage):
                y_loc[k][j] = float(-1)

            if len(nonZerosOutput) != 0 and currentTimeStepOutput[0] != 0.05:
                # Get activity and model gradient activation around it
                currentOutputLoc = pref[nonZerosOutput[0]]
                currentActivation_Output = add_x_loc(currentOutputLoc, pref) + 0.05 # adding noise
                unitRingOutput = np.around(unitRingOutput + currentActivation_Output, decimals=2)
                # Store
                currentFinalRow = np.concatenate((Output[i][j][0:1], unitRingOutput))
                Output[i][j][0:33] = currentFinalRow
                # Complete y_loc matrix
                for k in range(numFixStepsAverage, TotalStepsAverage):
                    y_loc[k][j] = pref[nonZerosOutput[0]]

    # Change dtype of every element in matrix to float32 for later validation functions
    for i in range(0, Output.shape[0]):
        for j in range(0, Output.shape[1]):
            for k in range(0, Output.shape[2]):
                Output[i][j][k] = np.float32(Output[i][j][k])
    # Also change dtype for entire array
    Output = Output.astype('float32')

    # Sanity check
    print('Output solved: ', df['Spreadsheet'][0], ' ', df['TimeLimit'][0])

    return Input, Output, y_loc

# RP & RP Anti & RP Ctx1 & RP Ctx2
def prepare_RP(file_location, sequence_on, sequence_off):
    # For bug fixing
    # file_location, sequence_on, sequence_off = os.getcwd() + '\\Data CSP\\JW\\7962306_RP_Anti_easy_1100.xlsx', 0, 48
    # Open .xlsx and select necessary columns
    # print(file_location)
    df = pd.read_excel(file_location, engine='openpyxl')
    # Add all necessary columns to create the Yang form later
    df.loc[:, 'Fixation input'] = 1 # means: should fixate, no response
    df.loc[:, 'Fixation output'] = 0.8 # means: should response, no fixation
    df.loc[:, 'DM'] = 0
    df.loc[:, 'DM Anti'] = 0
    df.loc[:, 'EF'] = 0
    df.loc[:, 'EF Anti'] = 0
    df.loc[:, 'RP'] = 0
    df.loc[:, 'RP Anti'] = 0
    df.loc[:, 'RP Ctx1'] = 0
    df.loc[:, 'RP Ctx2'] = 0
    df.loc[:, 'WM'] = 0
    df.loc[:, 'WM Anti'] = 0
    df.loc[:, 'WM Ctx1'] = 0
    df.loc[:, 'WM Ctx2'] = 0
    # Reorder columns
    df_selection = df[['Spreadsheet', 'TimeLimit', 'Onset Time', 'Object Name', 'Spreadsheet: CorrectAnswer1', 'Correct', 'Component Name', \
                       'Fixation input', 'Fixation output', 'Spreadsheet: Field 1', 'Spreadsheet: Field 2', 'Spreadsheet: Field 3', 'Spreadsheet: Field 4',\
                       'Spreadsheet: Field 5', 'Spreadsheet: Field 6', 'Spreadsheet: Field 7', 'Spreadsheet: Field 8', 'Spreadsheet: Field 9',\
                       'Spreadsheet: Field 10', 'Spreadsheet: Field 11', 'Spreadsheet: Field 12', 'Spreadsheet: Field 13', 'Spreadsheet: Field 14',\
                       'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17', 'Spreadsheet: Field 18', 'Spreadsheet: Field 19',\
                       'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22', 'Spreadsheet: Field 23', 'Spreadsheet: Field 24',\
                       'Spreadsheet: Field 25', 'Spreadsheet: Field 26', 'Spreadsheet: Field 27', 'Spreadsheet: Field 28', 'Spreadsheet: Field 29',\
                       'Spreadsheet: Field 30', 'Spreadsheet: Field 31', 'Spreadsheet: Field 32', \
                       'Spreadsheet: Field 1', 'Spreadsheet: Field 2', 'Spreadsheet: Field 3', 'Spreadsheet: Field 4', \
                       'Spreadsheet: Field 5', 'Spreadsheet: Field 6', 'Spreadsheet: Field 7', 'Spreadsheet: Field 8', 'Spreadsheet: Field 9', \
                       'Spreadsheet: Field 10', 'Spreadsheet: Field 11', 'Spreadsheet: Field 12', 'Spreadsheet: Field 13', 'Spreadsheet: Field 14',\
                       'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17', 'Spreadsheet: Field 18', 'Spreadsheet: Field 19',\
                       'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22', 'Spreadsheet: Field 23',\
                       'Spreadsheet: Field 24', 'Spreadsheet: Field 25', 'Spreadsheet: Field 26', 'Spreadsheet: Field 27', 'Spreadsheet: Field 28',\
                       'Spreadsheet: Field 29', 'Spreadsheet: Field 30', 'Spreadsheet: Field 31', 'Spreadsheet: Field 32', 'DM',\
                       'DM Anti', 'EF', 'EF Anti', 'RP', 'RP Anti', 'RP Ctx1', 'RP Ctx2', 'WM', 'WM Anti', 'WM Ctx1', 'WM Ctx1', 'Spreadsheet: display']]
    # Prepare incrementation for trials between fixation events in df
    incrementList = []
    for i in range(0,len(df_selection)):
        if df_selection['Component Name'][i] == 'Fixation' and df_selection['Component Name'][i+1] != 'Fixation'\
                and df_selection['Object Name'][i+1] != 'Response': # Get all rows with that string as they start a trial sequence
            incrementList.append(i)

    # Get average epoch time steps for the selected task in one session
    finalTrialsList = []
    numFixStepsTotal = 0
    numRespStepsTotal = 0
    iterationSteps = 0
    for i in incrementList:
        currentTrial = df_selection[i:i + 2].reset_index().drop(columns=['index'])
        iterationSteps = iterationSteps + 1
        numFixSteps = round(currentTrial['Onset Time'][0]/20) # equal to neuronal time constant of 20ms (Yang, 2019)
        numFixStepsTotal = numFixStepsTotal + numFixSteps
        numRespSteps = round(currentTrial['Onset Time'][1]/20) # equal to neuronal time constant of 20ms (Yang, 2019)
        numRespStepsTotal = numRespStepsTotal + numRespSteps

    # numFixStepsAverage = round(numFixStepsTotal/iterationSteps)
    # numRespStepsAverage = round(numRespStepsTotal/iterationSteps)
    # TotalStepsAverage = numFixStepsAverage + numRespStepsAverage

    numFixStepsAverage = 30
    numRespStepsAverage = 70
    TotalStepsAverage = numFixStepsAverage + numRespStepsAverage

    # Take all trials and high-sample them to the average steps
    for i in incrementList:
        currentTrial = df_selection[i+1:i+2].reset_index().drop(columns=['index'])
        # print(currentTrial['Object Name'])
        # print(i)
        # Create List with high-sampled rows for both epochs
        currentSequenceList = []
        for i in range(0, numFixStepsAverage+numRespStepsAverage):
            sequence = [currentTrial.iloc[0]]
            currentSequenceList.append(sequence)
        # Append current trial to final list - corresponds to one batch/ one task in one session
        finalTrialsList.append(currentSequenceList)


    # Create final df for INPUT and OUPUT
    newOrderList = []
    # Append all the time steps accordingly to a list
    for j in range(0, TotalStepsAverage):
        for i in range(0, len(finalTrialsList)):
            newOrderList.append(finalTrialsList[i][j])

    # Create Yang form
    finalTrialsList_array = np.array(newOrderList).reshape((len(finalTrialsList[0]),len(finalTrialsList),86))
    # Create one input file and one output file
    Input = finalTrialsList_array
    Output = finalTrialsList_array
    # Create final Input form
    Input = np.delete(Input,[0,1,2,3,4,5,6,8,77],axis = 2)
    # Create final output form
    Output = np.delete(Output,np.s_[0,1,2,4,5,6,7],axis = 2)
    Output = np.delete(Output,np.s_[34:78],axis = 2)
    # Delete all rows that are not needed (for either training or testing)
    Input = Input[:, sequence_on:sequence_off, :]
    Output = Output[:, sequence_on:sequence_off, :]

    # INPUT ############################################################################################################
    # float all fixation input values to 1
    for i in range(0,Input.shape[0]):
        for j in range(0, Input.shape[1]):
            Input[i][j][0] = np.float32(1)
    # float all task values to 0
    for i in range(0,Input.shape[0]):
        for j in range(0, Input.shape[1]):
            for k in range(65,Input.shape[2]):
                Input[i][j][k] = np.float32(0)
    # Float current task value to 1
    taskDict = {
      'DM ': 65,
      'DM Anti ': 66,
      'EF ': 67,
      'EF Anti ': 68,
      'RP ': 69,
      'RP Anti ': 70,
      'RP Ctx1 ': 71,
      'RP Ctx2 ': 72,
      'WM ': 73,
      'WM Anti ': 74,
      'WM Ctx1 ': 75,
      'WM Ctx2 ': 76
    }
    # float task values to 1
    for i in range(0,Input.shape[0]):
        for j in range(0, Input.shape[1]):
            Input[i][j][taskDict[df['Spreadsheet'][0].split('-')[0]]] = np.float32(1)
    # float all NaN's on field units to 0
    for i in range(0,Input.shape[0]):
        for j in range(0, Input.shape[1]):
            for k in range(1,65):
                if Input[i][j][k] == 'NaN.png':
                    Input[i][j][k] = np.float32(0)
    # float all values on mod1 fields to their true value (arrow direction)
    mod1Dict = {
      '60_0': 0.08,
      '60_1': 0.17,
      '120_0': 0.25,
      '120_1': 0.33,
      '180_0': 0.42,
      '180_1': 0.5,
      '240_0': 0.58,
      '240_1': 0.66,
      '300_0': 0.75,
      '300_1': 0.83,
      '360_0': 0.92,
      '360_1': 1.0,
    }
    for i in range(0,Input.shape[0]):
        for j in range(0, Input.shape[1]):
            for k in range(1,33):
                if Input[i][j][k] != 0:
                    Input[i][j][k] = mod1Dict[Input[i][j][k].split('w')[0]]
    # float all values on mod2 fields to their true value (arrow strength)
    mod2Dict = {
      '0_25.png': np.float32(0.25),
      '0_50.png': np.float32(0.5),
      '0_75.png': np.float32(0.75),
      '1_0.png': np.float32(1.0),
    }
    for i in range(0,Input.shape[0]):
        for j in range(0, Input.shape[1]):
            for k in range(33,65):
                if Input[i][j][k] != 0:
                    Input[i][j][k] = mod2Dict[Input[i][j][k].split('w')[1]]

    # float all field values of fixation period to 0
    for i in range(0,numFixStepsAverage):
        for j in range(0, Input.shape[1]):
            for k in range(1,65):
                    Input[i][j][k] = np.float32(0)


    # Add input gradient activation
    # Create default hyperparameters for network
    num_ring, n_eachring, n_rule = 2, 32, 12
    n_input, n_output = 1 + num_ring * n_eachring + n_rule, n_eachring + 1
    pref = np.arange(0, 2 * np.pi, 2 * np.pi / 32)

    for i in range(0, Input.shape[0]):
        for j in range(0, Input.shape[1]):
            currentTimeStepModOne = Input[i][j][1:33]
            currentTimeStepModTwo = Input[i][j][33:65]
            # Allocate first unit ring
            unitRingMod1 = np.zeros(32, dtype='float32')
            unitRingMod2 = np.zeros(32, dtype='float32')

            # Get non-zero values of time steps on both modalities and also directly fix distance between units issue
            NonZero_Mod1 = np.nonzero(currentTimeStepModOne)[0]
            NonZero_Mod2 = np.nonzero(currentTimeStepModTwo)[0]
            if len(NonZero_Mod1) != 0:
                # Accumulating all activities for both unit rings together
                for k in range(0,len(NonZero_Mod1)):
                    currentStimLoc_Mod1 = pref[NonZero_Mod1[k]]
                    currentStimStrength_Mod1 = currentTimeStepModOne[NonZero_Mod1[k]]
                    currentStimLoc_Mod2 = pref[NonZero_Mod2[k]]
                    currentStimStrength_Mod2 = currentTimeStepModTwo[NonZero_Mod2[k]]
                    # add one gradual activated stim to final form
                    currentActivation_Mod1 = add_x_loc(currentStimLoc_Mod1, pref) * currentStimStrength_Mod1
                    currentActivation_Mod2 = add_x_loc(currentStimLoc_Mod2, pref) * currentStimStrength_Mod2
                    # Add all activations for one trial together
                    unitRingMod1 = np.around(unitRingMod1 + currentActivation_Mod1, decimals=2)
                    unitRingMod2 = np.around(unitRingMod2 + currentActivation_Mod2, decimals=2)
                # Store
                currentFinalRow = np.concatenate((Input[i][j][0:1], unitRingMod1, unitRingMod2, Input[i][j][65:78]))
                Input[i][j][0:78] = currentFinalRow

    # Change dtype of every element in matrix to float32 for later validation functions
    for i in range(0, Input.shape[0]):
        for j in range(0, Input.shape[1]):
            for k in range(0, Input.shape[2]):
                Input[i][j][k] = np.float32(Input[i][j][k])
    # Also change dtype for entire array
    Input = Input.astype('float32')

    # Sanity check
    print('Input solved: ', df['Spreadsheet'][0], ' ', df['TimeLimit'][0])

    # OUTPUT ###########################################################################################################
    # float all field units during fixation epoch on 0.05
    for i in range(0, numFixStepsAverage):
        for j in range(0, Output.shape[1]):
            for k in range(2, 34):
                Output[i][j][k] = np.float32(0.05)
    # float all field units of response epoch to 0
    for i in range(numFixStepsAverage,TotalStepsAverage):
        for j in range(0, Output.shape[1]):
            for k in range(2, 34):
                Output[i][j][k] = np.float32(0)
    # float all fixation outputs during response period to 0.05
    for i in range(numFixStepsAverage,TotalStepsAverage):
        for j in range(0, Output.shape[1]):
                Output[i][j][1] = np.float32(0.05)

    # Assign field units to their according participant response value after fixation period
    outputDict_RP_1 = {
        'Image 2': 2,
        'Image 4': 4,
        'Image 6': 6,
        'Image 8': 8,
        'Image 10': 10,
        'Image 12': 12,
        'Image 14': 14,
        'Image 16': 16,
        'Image 18': 18,
        'Image 20': 20,
        'Image 22': 22,
        'Image 24': 24,
        'Image 26': 26,
        'Image 28': 28,
        'Image 30': 30,
        'Image 32': 32
    }

    outputDict_RP_2 = {
        'Image 1': 1,
        'Image 3': 3,
        'Image 5': 5,
        'Image 7': 7,
        'Image 9': 9,
        'Image 11': 11,
        'Image 13': 13,
        'Image 15': 15,
        'Image 17': 17,
        'Image 19': 19,
        'Image 21': 21,
        'Image 23': 23,
        'Image 25': 25,
        'Image 27': 27,
        'Image 29': 29,
        'Image 31': 31,
    }

    for i in range(numFixStepsAverage,TotalStepsAverage):
        for j in range(0, Output.shape[1]):
            # Get the right dictionary
            if Output[i][j][34].split(' RP')[0] == 'Display 1':
                outputDict = outputDict_RP_1
            elif Output[i][j][34].split(' RP')[0] == 'Display 2':
                outputDict = outputDict_RP_2

            if Output[i][j][0] != 'screen': # and Output[i][j][0] != 'object-2333' and Output[i][j][0] != 'object-2330': # todo: Why needs RP easy 1100 this bug fix?
                Output[i][j][outputDict[Output[i][j][0]]] = np.float32(0.85)
            else:
                for k in range(2,34):
                    Output[i][j][k] = np.float32(0.05)

    # Drop unnecessary first column
    Output = np.delete(Output,[0,34],axis = 2)
    # Pre-allocate y-loc matrix; needed for later validation
    y_loc = np.zeros((Output.shape[0], Output.shape[1]))

    # Add output gradient activation
    for i in range(0, Output.shape[0]):
        for j in range(0, Output.shape[1]):
            currentTimeStepOutput = Output[i][j][1:33]
            # Allocate first unit ring
            unitRingOutput = np.zeros(32, dtype='float32')
            # Get non-zero values of time steps
            nonZerosOutput = np.nonzero(currentTimeStepOutput)[0]

            # Float first fixations rows with -1 for validation matrix y-loc
            for k in range(0, numFixStepsAverage):
                y_loc[k][j] = np.float32(-1)

            if len(nonZerosOutput) != 0 and currentTimeStepOutput[0] != 0.05:
                # Get activity and model gradient activation around it
                currentOutputLoc = pref[nonZerosOutput[0]]
                currentActivation_Output = add_x_loc(currentOutputLoc, pref) + 0.05 # adding noise
                unitRingOutput = np.around(unitRingOutput + currentActivation_Output, decimals=2)
                # Store
                currentFinalRow = np.concatenate((Output[i][j][0:1], unitRingOutput))
                Output[i][j][0:33] = currentFinalRow
                # Complete y_loc matrix
                for k in range(numFixStepsAverage, TotalStepsAverage):
                    y_loc[k][j] = pref[nonZerosOutput[0]]

    # Change dtype of every element in matrix to float32 for later validation functions
    for i in range(0, Output.shape[0]):
        for j in range(0, Output.shape[1]):
            for k in range(0, Output.shape[2]):
                Output[i][j][k] = np.float32(Output[i][j][k])
    # Also change dtype for entire array
    Output = Output.astype('float32')

    # Sanity check
    print('Output solved: ', df['Spreadsheet'][0], ' ', df['TimeLimit'][0])

    return Input, Output, y_loc

# WM & WM Anti & WM Ctx1 & WM Ctx2
def prepare_WM(file_location, sequence_on, sequence_off):
    # For bug fixing
    # file_location, sequence_on, sequence_off = os.getcwd() + '\\Data CSP\\SC\\7962396_WM_Ctx1_hard_1300.xlsx', 0, 48
    # Open .xlsx and select necessary columns
    # print(file_location)
    df = pd.read_excel(file_location, engine='openpyxl')
    # Add all necessary columns to create the Yang form later
    df.loc[:, 'Fixation input'] = 1 # means: should fixate, no response
    df.loc[:, 'Fixation output'] = 0.8 # means: should response, no fixation
    df.loc[:, 'DM'] = 0
    df.loc[:, 'DM Anti'] = 0
    df.loc[:, 'EF'] = 0
    df.loc[:, 'EF Anti'] = 0
    df.loc[:, 'RP'] = 0
    df.loc[:, 'RP Anti'] = 0
    df.loc[:, 'RP Ctx1'] = 0
    df.loc[:, 'RP Ctx2'] = 0
    df.loc[:, 'WM'] = 0
    df.loc[:, 'WM Anti'] = 0
    df.loc[:, 'WM Ctx1'] = 0
    df.loc[:, 'WM Ctx2'] = 0
    # Reorder columns
    df_selection = df[['Spreadsheet', 'TimeLimit', 'Onset Time', 'Object Name', 'Object ID', 'Spreadsheet: CorrectAnswer', 'Correct', 'Component Name', \
                       'Fixation input', 'Fixation output', 'Spreadsheet: Field 1', 'Spreadsheet: Field 2', 'Spreadsheet: Field 3', 'Spreadsheet: Field 4',\
                       'Spreadsheet: Field 5', 'Spreadsheet: Field 6', 'Spreadsheet: Field 7', 'Spreadsheet: Field 8', 'Spreadsheet: Field 9',\
                       'Spreadsheet: Field 10', 'Spreadsheet: Field 11', 'Spreadsheet: Field 12', 'Spreadsheet: Field 13', 'Spreadsheet: Field 14',\
                       'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17', 'Spreadsheet: Field 18', 'Spreadsheet: Field 19',\
                       'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22', 'Spreadsheet: Field 23', 'Spreadsheet: Field 24',\
                       'Spreadsheet: Field 25', 'Spreadsheet: Field 26', 'Spreadsheet: Field 27', 'Spreadsheet: Field 28', 'Spreadsheet: Field 29',\
                       'Spreadsheet: Field 30', 'Spreadsheet: Field 31', 'Spreadsheet: Field 32', \
                       'Spreadsheet: Field 1', 'Spreadsheet: Field 2', 'Spreadsheet: Field 3', 'Spreadsheet: Field 4', \
                       'Spreadsheet: Field 5', 'Spreadsheet: Field 6', 'Spreadsheet: Field 7', 'Spreadsheet: Field 8', 'Spreadsheet: Field 9', \
                       'Spreadsheet: Field 10', 'Spreadsheet: Field 11', 'Spreadsheet: Field 12', 'Spreadsheet: Field 13', 'Spreadsheet: Field 14',\
                       'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17', 'Spreadsheet: Field 18', 'Spreadsheet: Field 19',\
                       'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22', 'Spreadsheet: Field 23',\
                       'Spreadsheet: Field 24', 'Spreadsheet: Field 25', 'Spreadsheet: Field 26', 'Spreadsheet: Field 27', 'Spreadsheet: Field 28',\
                       'Spreadsheet: Field 29', 'Spreadsheet: Field 30', 'Spreadsheet: Field 31', 'Spreadsheet: Field 32', 'DM',\
                       'DM Anti', 'EF', 'EF Anti', 'RP', 'RP Anti', 'RP Ctx1', 'RP Ctx2', 'WM', 'WM Anti', 'WM Ctx1', 'WM Ctx1', 'Spreadsheet: display']]
    # Prepare incrementation for trials between fixation events in df
    incrementList = []
    for i in range(0,len(df_selection)):
        if df_selection['Component Name'][i] == 'Fixation': # Get all rows with that string as they start a trial sequence
            incrementList.append(i)

    # Get average epoch time steps for the selected task in one session
    finalTrialsList = []
    numFixStepsTotal = 0
    numStimStepsTotal = 0
    numDelayStepsTotal = 0
    numRespStepsTotal = 0
    iterationSteps = 1
    for i in incrementList:
        if iterationSteps == len(incrementList):
            break
        currentTrial = df_selection[i:i+2].reset_index().drop(columns=['index'])
        consecutiveTrial = df_selection[incrementList[iterationSteps]:incrementList[iterationSteps]+2].reset_index().drop(columns=['index'])
        iterationSteps += 1


        ################################################################################################################
        # todo: Fixation Cross 1
        numFixSteps = round(currentTrial['Onset Time'][0]/20) # equal to neuronal time constant of 20ms (Yang, 2019)
        numFixStepsTotal = numFixStepsTotal + numFixSteps
        # todo: Stim presentation 1
        numStimSteps = round(currentTrial['Onset Time'][1]/20)  # equal to neuronal time constant of 20ms (Yang, 2019)
        numStimStepsTotal = numStimStepsTotal + numStimSteps
        # todo: Delay 1 = Fixation Cross 2
        numDelaySteps = round(consecutiveTrial['Onset Time'][0]/20)  # equal to neuronal time constant of 20ms (Yang, 2019)
        numDelayStepsTotal = numDelayStepsTotal + numDelaySteps
        # todo: Response 1 = Stim presentation 2
        numRespSteps = round(consecutiveTrial['Onset Time'][1]/20)  # equal to neuronal time constant of 20ms (Yang, 2019)
        numRespStepsTotal = numRespStepsTotal + numRespSteps
        ################################################################################################################


    # numFixStepsAverage = round(numFixStepsTotal/iterationSteps)
    # numStimStepsAverage = round(numStimStepsTotal/iterationSteps)
    # numDelayStepsAverage = round(numDelayStepsTotal/iterationSteps)
    # numRespStepsAverage = round(numRespStepsTotal/iterationSteps)
    # TotalStepsAverage = numFixStepsAverage + numStimStepsAverage + numDelayStepsAverage + numRespStepsAverage

    # todo: For bug fixing we will create equal sequence length for all sessions
    numFixStepsAverage = 20
    numStimStepsAverage = 40
    numDelayStepsAverage = 20
    numRespStepsAverage = 20
    TotalStepsAverage = numFixStepsAverage + numStimStepsAverage + numDelayStepsAverage + numRespStepsAverage

    # Take all trials and high-sample them to the average steps
    incrementSteps = 1
    for i in incrementList:
        if incrementSteps == len(incrementList):
            break
        currentTrial = df_selection[i+1:i+2].reset_index().drop(columns=['index'])
        consecutiveTrial = df_selection[incrementList[incrementSteps]+1:incrementList[incrementSteps]+2].reset_index().drop(columns=['index'])
        incrementSteps += 1

        # Create List with high-sampled rows for first 2 epochs
        currentSequenceList = []
        for j in range(0, numFixStepsAverage+numStimStepsAverage):
            sequence = [currentTrial.iloc[0]]
            currentSequenceList.append(sequence)
        # And for second 2 epochs
        for k in range(0, numDelayStepsAverage+numRespStepsAverage):
            sequence = [consecutiveTrial.iloc[0]]
            currentSequenceList.append(sequence)
        # Append current trial to final list - corresponds to one batch/ one task in one session
        finalTrialsList.append(currentSequenceList)


    # Create final df for INPUT and OUPUT
    newOrderList = []
    # Append all the time steps accordingly to a list
    for j in range(0, TotalStepsAverage):
        for i in range(0, len(finalTrialsList)):
            newOrderList.append(finalTrialsList[i][j])

    # Create Yang form
    finalTrialsList_array = np.array(newOrderList).reshape((len(finalTrialsList[0]),len(finalTrialsList),87))
    # Create one input file and one output file
    Input = finalTrialsList_array
    Output = finalTrialsList_array
    # Create final Input form
    Input = np.delete(Input,[0,1,2,3,4,5,6,7,9,77],axis = 2)
    # Create final output form
    Output = np.delete(Output,np.s_[0,1,2,5,6,7,8],axis = 2)
    Output = np.delete(Output,np.s_[34:78],axis = 2)
    # Delete all rows that are not needed (for either training or testing)
    Input = Input[:, sequence_on:sequence_off, :]
    Output = Output[:, sequence_on:sequence_off, :]

    # INPUT ############################################################################################################
    # float all fixation input values to 1
    for i in range(0,Input.shape[0]):
        for j in range(0,Input.shape[1]):
            Input[i][j][0] = np.float32(1)
    # float all task values to 0
    for i in range(0,Input.shape[0]):
        for j in range(0,Input.shape[1]):
            for k in range(65,Input.shape[2]):
                Input[i][j][k] = np.float32(0)
    # Float current task value to 1
    taskDict = {
      'DM ': 65,
      'DM Anti ': 66,
      'EF ': 67,
      'EF Anti ': 68,
      'RP ': 69,
      'RP Anti ': 70,
      'RP Ctx1 ': 71,
      'RP Ctx2 ': 72,
      'WM ': 73,
      'WM Anti ': 74,
      'WM Ctx1 ': 75,
      'WM Ctx2 ': 76
    }
    # float task values to 1
    for i in range(0,Input.shape[0]):
        for j in range(0,Input.shape[1]):
            Input[i][j][taskDict[df['Spreadsheet'][0].split('-')[0]]] = np.float32(1)
    # float all NaN's on field units to 0
    for i in range(0,Input.shape[0]):
        for j in range(0,Input.shape[1]):
            for k in range(1,65):
                if Input[i][j][k] == 'NaN.png' or pd.isna(Input[i][j][k]):
                    Input[i][j][k] = np.float32(0)
    # float all values on mod1 fields to their true value
    mod1Dict = {
      '60_0': 0.08,
      '60_1': 0.17,
      '120_0': 0.25,
      '120_1': 0.33,
      '180_0': 0.42,
      '180_1': 0.5,
      '240_0': 0.58,
      '240_1': 0.66,
      '300_0': 0.75,
      '300_1': 0.83,
      '360_0': 0.92,
      '360_1': 1.0,
    }
    for i in range(0,Input.shape[0]):
        for j in range(0,Input.shape[1]):
            for k in range(1,33):
                if Input[i][j][k] != 0:
                    Input[i][j][k] = mod1Dict[Input[i][j][k].split('w')[0]]
    # float all values on mod2 fields to their true value
    mod2Dict = {
      '0_25.png': np.float32(0.25),
      '0_50.png': np.float32(0.5),
      '0_75.png': np.float32(0.75),
      '1_0.png': np.float32(1.0),
    }
    for i in range(0,Input.shape[0]):
        for j in range(0,Input.shape[1]):
            for k in range(33,65):
                if Input[i][j][k] != 0:
                    Input[i][j][k] = mod2Dict[Input[i][j][k].split('w')[1]]

    # float all field values of fixation period to 0
    for i in range(0,numFixStepsAverage):
        for j in range(0,Input.shape[1]):
            for k in range(1,65):
                    Input[i][j][k] = np.float32(0)


    # Add input gradient activation
    # Create default hyperparameters for network
    num_ring, n_eachring, n_rule = 2, 32, 12
    n_input, n_output = 1 + num_ring * n_eachring + n_rule, n_eachring + 1
    pref = np.arange(0, 2 * np.pi, 2 * np.pi / 32)

    for i in range(0, Input.shape[0]):
        for j in range(0,Input.shape[1]):
            currentTimeStepModOne = Input[i][j][1:33]
            currentTimeStepModTwo = Input[i][j][33:65]
            # Allocate first unit ring
            unitRingMod1 = np.zeros(32, dtype='float32')
            unitRingMod2 = np.zeros(32, dtype='float32')

            # Get non-zero values of time steps on both modalities and also directly fix distance between units issue
            NonZero_Mod1 = np.nonzero(currentTimeStepModOne)[0]
            NonZero_Mod2 = np.nonzero(currentTimeStepModTwo)[0]
            if len(NonZero_Mod1) != 0:
                # Accumulating all activities for both unit rings together
                for k in range(0,len(NonZero_Mod1)):
                    currentStimLoc_Mod1 = pref[NonZero_Mod1[k]]
                    currentStimStrength_Mod1 = currentTimeStepModOne[NonZero_Mod1[k]]
                    currentStimLoc_Mod2 = pref[NonZero_Mod2[k]]
                    currentStimStrength_Mod2 = currentTimeStepModTwo[NonZero_Mod2[k]]
                    # add one gradual activated stim to final form
                    currentActivation_Mod1 = add_x_loc(currentStimLoc_Mod1, pref) * currentStimStrength_Mod1
                    currentActivation_Mod2 = add_x_loc(currentStimLoc_Mod2, pref) * currentStimStrength_Mod2
                    # Add all activations for one trial together
                    unitRingMod1 = np.around(unitRingMod1 + currentActivation_Mod1, decimals=2)
                    unitRingMod2 = np.around(unitRingMod2 + currentActivation_Mod2, decimals=2)
                # Store
                currentFinalRow = np.concatenate((Input[i][j][0:1], unitRingMod1, unitRingMod2, Input[i][j][65:78]))
                Input[i][j][0:78] = currentFinalRow

    # Change dtype of every element in matrix to float32 for later validation functions
    for i in range(0, Input.shape[0]):
        for j in range(0, Input.shape[1]):
            for k in range(0, Input.shape[2]):
                Input[i][j][k] = np.float32(Input[i][j][k])
    # Also change dtype for entire array
    Input = Input.astype('float32')

    # Sanity check
    print('Input solved: ', df['Spreadsheet'][0], ' ', df['TimeLimit'][0])

    # OUTPUT ###########################################################################################################
    # float all field units during fixation epoch on 0.05
    for i in range(0, numFixStepsAverage+numStimStepsAverage+numDelayStepsAverage):
        for j in range(0,Output.shape[1]):
            for k in range(3, 35):
                Output[i][j][k] = np.float32(0.05)
    # float all field units of response epoch to 0
    for i in range(numFixStepsAverage+numStimStepsAverage+numDelayStepsAverage,TotalStepsAverage):
        for j in range(0,Output.shape[1]):
            for k in range(3, 35):
                Output[i][j][k] = np.float32(0)
    # float all fixation outputs during response period to 0.05
    for i in range(numFixStepsAverage+numStimStepsAverage+numDelayStepsAverage,TotalStepsAverage):
        for j in range(0,Output.shape[1]):
                Output[i][j][2] = np.float32(0.05)

    # Assign field units to their according participant response value after fixation period
    outputDict_WM = {
        'Image 1': 1,
        'Image 2': 2,
        'Image 3': 3,
        'Image 4': 4,
        'Image 5': 5,
        'Image 6': 6,
        'Image 7': 7,
        'Image 8': 8,
        'Image 9': 9,
        'Image 10': 10,
        'Image 11': 11,
        'Image 12': 12,
        'Image 13': 13,
        'Image 14': 14,
        'Image 15': 15,
        'Image 16': 16,
        'Image 17': 17,
        'Image 18': 18,
        'Image 19': 19,
        'Image 20': 20,
        'Image 21': 21,
        'Image 22': 22,
        'Image 23': 23,
        'Image 24': 24,
        'Image 25': 25,
        'Image 26': 26,
        'Image 27': 27,
        'Image 28': 28,
        'Image 29': 29,
        'Image 30': 30,
        'Image 31': 31,
        'Image 32': 32
    }

    outputDict_WM_Ctx = {
        'object-1591': 8,
        'object-1593': 8,
        'object-1595': 8,
        'object-1597': 8,
        'object-1592': 24,
        'object-1594': 24,
        'object-1596': 24,
        'object-1598': 24,
    }


    for i in range(numFixStepsAverage+numStimStepsAverage+numDelayStepsAverage,TotalStepsAverage):
        for j in range(0,Output.shape[1]):
            if isinstance(Output[i][j][35], str):
                # Get the right dictionary
                if file_location.split('_')[3] != 'Ctx1' and file_location.split('_')[3] != 'Ctx2':
                    outputDict = outputDict_WM
                    chosenColumn = 0
                else:
                    outputDict = outputDict_WM_Ctx
                    chosenColumn = 1

                if Output[i][j][1] != 'screen' and Output[i][j][1] != 'Fixation Cross' and Output[i][j][1] != 'Response':
                    Output[i][j][outputDict[Output[i][j][chosenColumn]]] = np.float32(0.85)
                else:
                    for k in range(3,35):
                        Output[i][j][k] = np.float32(0.05)
            else:
                for k in range(3, 35):
                    Output[i][j][k] = np.float32(0.05)

    # Drop unnecessary columns
    Output = np.delete(Output,[0,1,35],axis = 2)
    # Pre-allocate y-loc matrix; needed for later validation
    y_loc = np.zeros((Output.shape[0], Output.shape[1]), dtype = 'float32')

    # Add output gradient activation
    for i in range(0, Output.shape[0]):
        for j in range(0,Output.shape[1]):
            currentTimeStepOutput = Output[i][j][1:33]
            # Allocate first unit ring
            unitRingOutput = np.zeros(32, dtype='float32')
            # Get non-zero values of time steps
            nonZerosOutput = np.nonzero(currentTimeStepOutput)[0]

            # Float first fixations rows with -1 for validation matrix y-loc
            for k in range(0, numFixStepsAverage):
                y_loc[k][j] = np.float32(-1)

            if len(nonZerosOutput) != 0 and currentTimeStepOutput[0] != 0.05:
                # Get activity and model gradient activation around it
                currentOutputLoc = pref[nonZerosOutput[0]]
                currentActivation_Output = add_x_loc(currentOutputLoc, pref) + 0.05 # adding noise
                unitRingOutput = np.around(unitRingOutput + currentActivation_Output, decimals=2)
                # Store
                currentFinalRow = np.concatenate((Output[i][j][0:1], unitRingOutput))
                Output[i][j][0:33] = currentFinalRow
                # Complete y_loc matrix
                for k in range(numFixStepsAverage, TotalStepsAverage):
                    y_loc[k][j] = pref[nonZerosOutput[0]]

    # Change dtype of every element in matrix to float32 for later validation functions
    for i in range(0, Output.shape[0]):
        for j in range(0, Output.shape[1]):
            for k in range(0, Output.shape[2]):
                Output[i][j][k] = np.float32(Output[i][j][k])
    # Also change dtype for entire array
    Output = Output.astype('float32')

    # Sanity check
    print('Output solved: ', df['Spreadsheet'][0], ' ', df['TimeLimit'][0])

    return Input, Output, y_loc


########################################################################################################################
# co: Step 2: Create lists with files for preperation functions ########################################################
########################################################################################################################
# General .xlsx list
xlsxFolderList = os.listdir(os.getcwd() + '\\Data CSP\\')

def fileDict(xlsxFolder, xlsxFolderList):
    # Create file dictionary
    file_dict = dict()
    # Allocate lists for every task
    file_dict['filesList_DM_easy'], file_dict['filesList_DM_normal'], file_dict['filesList_DM_hard'] = [], [], []
    file_dict['filesList_DMAnti_easy'], file_dict['filesList_DMAnti_normal'], file_dict['filesList_DMAnti_hard'] = [], [], []
    file_dict['filesList_EF_easy'], file_dict['filesList_EF_normal'], file_dict['filesList_EF_hard'] = [], [], []
    file_dict['filesList_EFAnti_easy'], file_dict['filesList_EFAnti_normal'], file_dict['filesList_EFAnti_hard'] = [], [], []
    file_dict['filesList_RP_easy'], file_dict['filesList_RP_normal'], file_dict['filesList_RP_hard'] = [], [], []
    file_dict['filesList_RPAnti_easy'], file_dict['filesList_RPAnti_normal'], file_dict['filesList_RPAnti_hard'] = [], [], []
    file_dict['filesList_RPCtx1_easy'], file_dict['filesList_RPCtx1_normal'], file_dict['filesList_RPCtx1_hard'] = [], [], []
    file_dict['filesList_RPCtx2_easy'], file_dict['filesList_RPCtx2_normal'], file_dict['filesList_RPCtx2_hard'] = [], [], []
    file_dict['filesList_WM_easy'], file_dict['filesList_WM_normal'], file_dict['filesList_WM_hard'] = [], [], []
    file_dict['filesList_WMAnti_easy'], file_dict['filesList_WMAnti_normal'], file_dict['filesList_WMAnti_hard'] = [], [], []
    file_dict['filesList_WMCtx1_easy'], file_dict['filesList_WMCtx1_normal'], file_dict['filesList_WMCtx1_hard'] = [], [], []
    file_dict['filesList_WMCtx2_easy'], file_dict['filesList_WMCtx2_normal'], file_dict['filesList_WMCtx2_hard'] = [], [], []
    # Create row dictionary
    row_dict = dict()
    # Define row indices for storing the right .xlsx into file lists
    row_dict['filesList_DM_easy'], row_dict['filesList_DM_normal'], row_dict['filesList_DM_hard'] = [9,12], [15,18], [12,15]
    row_dict['filesList_DMAnti_easy'], row_dict['filesList_DMAnti_normal'], row_dict['filesList_DMAnti_hard'] = [0,3], [6,9], [3,6]
    row_dict['filesList_EF_easy'], row_dict['filesList_EF_normal'], row_dict['filesList_EF_hard'] = [27,30], [33,36], [30,33]
    row_dict['filesList_EFAnti_easy'], row_dict['filesList_EFAnti_normal'], row_dict['filesList_EFAnti_hard'] = [18,21], [24,27], [21,24]
    row_dict['filesList_RP_easy'], row_dict['filesList_RP_normal'], row_dict['filesList_RP_hard'] = [63,66], [69,72], [66,69]
    row_dict['filesList_RPAnti_easy'], row_dict['filesList_RPAnti_normal'], row_dict['filesList_RPAnti_hard'] = [36,39], [42,45], [39,42]
    row_dict['filesList_RPCtx1_easy'], row_dict['filesList_RPCtx1_normal'], row_dict['filesList_RPCtx1_hard'] = [45,48], [51,54], [48,51]
    row_dict['filesList_RPCtx2_easy'], row_dict['filesList_RPCtx2_normal'], row_dict['filesList_RPCtx2_hard'] = [54,57], [60,63], [57,60]
    row_dict['filesList_WM_easy'], row_dict['filesList_WM_normal'], row_dict['filesList_WM_hard'] = [99,102], [105,108], [102,105]
    row_dict['filesList_WMAnti_easy'], row_dict['filesList_WMAnti_normal'], row_dict['filesList_WMAnti_hard'] = [72,75], [78,81], [75,78]
    row_dict['filesList_WMCtx1_easy'], row_dict['filesList_WMCtx1_normal'], row_dict['filesList_WMCtx1_hard'] = [81,84], [87,90], [84,87]
    row_dict['filesList_WMCtx2_easy'], row_dict['filesList_WMCtx2_normal'], row_dict['filesList_WMCtx2_hard'] = [90,93], [96,99], [93,96]

    # Fill list for different task difficulties over all participants
    for j in file_dict:
        for i in xlsxFolderList[1:7]:
            xlsxFileList = os.listdir(xlsxFolder + i)
            # Fill all lists
            for k in range(row_dict[j][0],row_dict[j][1]):
                file_location = xlsxFolder + i + '/' + xlsxFileList[k]
                file_dict[j].append(file_location)

    # Create final lists
    # Append all DM
    AllDM_list = [*file_dict['filesList_DM_easy'], *file_dict['filesList_DM_normal'], *file_dict['filesList_DM_hard'],\
                  *file_dict['filesList_DMAnti_easy'], *file_dict['filesList_DMAnti_normal'], *file_dict['filesList_DMAnti_hard']]
    # Append all EF
    AllEF_list = [*file_dict['filesList_EF_easy'], *file_dict['filesList_EF_normal'], *file_dict['filesList_EF_hard'], \
                  *file_dict['filesList_EFAnti_easy'], *file_dict['filesList_EFAnti_normal'], *file_dict['filesList_EFAnti_hard']]
    # Append all RP
    AllRP_list = [*file_dict['filesList_RP_easy'], *file_dict['filesList_RP_normal'], *file_dict['filesList_RP_hard'], \
                  *file_dict['filesList_RPAnti_easy'], *file_dict['filesList_RPAnti_normal'], *file_dict['filesList_RPAnti_hard'],\
                  *file_dict['filesList_RPCtx1_easy'], *file_dict['filesList_RPCtx1_normal'], *file_dict['filesList_RPCtx1_hard'],\
                  *file_dict['filesList_RPCtx2_easy'], *file_dict['filesList_RPCtx2_normal'], *file_dict['filesList_RPCtx2_hard']]
    # Append all WM
    AllWM_list = [*file_dict['filesList_WM_easy'], *file_dict['filesList_WM_normal'], *file_dict['filesList_WM_hard'], \
                  *file_dict['filesList_WMAnti_easy'], *file_dict['filesList_WMAnti_normal'], *file_dict['filesList_WMAnti_hard'], \
                  *file_dict['filesList_WMCtx1_easy'], *file_dict['filesList_WMCtx1_normal'], *file_dict['filesList_WMCtx1_hard'], \
                  *file_dict['filesList_WMCtx2_easy'], *file_dict['filesList_WMCtx2_normal'], *file_dict['filesList_WMCtx2_hard']]

    # Append all tasks
    AllTasks_list = [*AllDM_list, *AllEF_list, *AllRP_list, * AllWM_list]

    return AllTasks_list


# ########################################################################################################################
# # co: Step 3: Train network with batches from filesList ################################################################
# ########################################################################################################################
# # DM batches todo: All batches can be treated equally
# # corresponds 9 batches of 60 sequences, respectively, on easy level
# for i in file_dict['filesList_DM_easy']:
#     Input, Output = prepare_DM(i)
#     # print(Input, Output)
# # corresponds 9 batches of 60 sequences, respectively, on normal level
# for i in file_dict['filesList_DM_normal']:
#     Input, Output = prepare_DM(i)
#     # print(Input, Output)
# # corresponds 9 batches of 60 sequences, respectively, on hard level
# for i in file_dict['filesList_DM_hard']:
#     Input, Output = prepare_DM(i)
#     # print(Input, Output)
#
# # DM Anti batches todo: All batches can be treated equally
# # corresponds 9 batches of 60 sequences, respectively, on easy level
# for i in file_dict['filesList_DMAnti_easy']:
#     Input, Output = prepare_DM(i)
#     # print(Input, Output)
# # corresponds 9 batches of 60 sequences, respectively, on normal level
# for i in file_dict['filesList_DMAnti_normal']:
#     Input, Output = prepare_DM(i)
#     # print(Input, Output)
# # corresponds 9 batches of 60 sequences, respectively, on hard level
# for i in file_dict['filesList_DMAnti_hard']:
#     Input, Output = prepare_DM(i)
#     # print(Input, Output)
#
# ########################################################################################################################
#
# # EF batches todo: All batches can be treated equally
# # corresponds 9 batches of 60 sequences, respectively, on easy level
# for i in file_dict['filesList_EF_easy']:
#     Input, Output = prepare_EF(i)
#     # print(Input, Output)
# # corresponds 9 batches of 60 sequences, respectively, on normal level
# for i in file_dict['filesList_EF_normal']:
#     Input, Output = prepare_EF(i)
#     # print(Input, Output)
# # corresponds 9 batches of 60 sequences, respectively, on hard level
# for i in file_dict['filesList_EF_hard']:
#     Input, Output = prepare_EF(i)
#     # print(Input, Output)
#
# # EF Anti batches todo: All batches can be treated equally
# # corresponds 9 batches of 60 sequences, respectively, on easy level
# for i in file_dict['filesList_EFAnti_easy']:
#     Input, Output = prepare_EF(i)
#     # print(Input, Output)
# # corresponds 9 batches of 60 sequences, respectively, on normal level
# for i in file_dict['filesList_EFAnti_normal']:
#     Input, Output = prepare_EF(i)
#     # print(Input, Output)
# # corresponds 9 batches of 60 sequences, respectively, on hard level
# for i in file_dict['filesList_EFAnti_hard']:
#     Input, Output = prepare_EF(i)
#     # print(Input, Output)
#
# ########################################################################################################################
#
# # RP batches todo: All batches can be treated equally
# # corresponds 9 batches of 60 sequences, respectively, on easy level
# for i in file_dict['filesList_RP_easy']:
#     Input, Output = prepare_RP(i)
#     # print(Input, Output)
# # corresponds 9 batches of 60 sequences, respectively, on normal level
# for i in file_dict['filesList_RP_normal']:
#     Input, Output = prepare_RP(i)
#     # print(Input, Output)
# # corresponds 9 batches of 60 sequences, respectively, on hard level
# for i in file_dict['filesList_RP_hard']:
#     Input, Output = prepare_RP(i)
#     # print(Input, Output)
#
# # RP Anti batches todo: All batches can be treated equally
# # corresponds 9 batches of 60 sequences, respectively, on easy level
# for i in file_dict['filesList_RPAnti_easy']:
#     Input, Output = prepare_RP(i)
#     # print(Input, Output)
# # corresponds 9 batches of 60 sequences, respectively, on normal level
# for i in file_dict['filesList_RPAnti_normal']:
#     Input, Output = prepare_RP(i)
#     # print(Input, Output)
# # corresponds 9 batches of 60 sequences, respectively, on hard level
# for i in file_dict['filesList_RPAnti_hard']:
#     Input, Output = prepare_RP(i)
#     # print(Input, Output)
#
# # RP Ctx1 batches todo: All batches can be treated equally
# # corresponds 9 batches of 60 sequences, respectively, on easy level
# for i in file_dict['filesList_RPCtx1_easy']:
#     Input, Output = prepare_RP(i)
#     # print(Input, Output)
# # corresponds 9 batches of 60 sequences, respectively, on normal level
# for i in file_dict['filesList_RPCtx1_normal']:
#     Input, Output = prepare_RP(i)
#     # print(Input, Output)
# # corresponds 9 batches of 60 sequences, respectively, on hard level
# for i in file_dict['filesList_RPCtx1_hard']:
#     Input, Output = prepare_RP(i)
#     # print(Input, Output)
#
# # RP Ctx2 batches todo: All batches can be treated equally
# # corresponds 9 batches of 60 sequences, respectively, on easy level
# for i in file_dict['filesList_RPCtx2_easy']:
#     Input, Output = prepare_RP(i)
#     # print(Input, Output)
# # corresponds 9 batches of 60 sequences, respectively, on normal level
# for i in file_dict['filesList_RPCtx2_normal']:
#     Input, Output = prepare_RP(i)
#     # print(Input, Output)
# # corresponds 9 batches of 60 sequences, respectively, on hard level
# for i in file_dict['filesList_RPCtx2_hard']:
#     Input, Output = prepare_RP(i)
#     # print(Input, Output)
#
# ########################################################################################################################
#
# # WM batches todo: All batches can be treated equally
# # corresponds 9 batches of 60 sequences, respectively, on easy level
# for i in file_dict['filesList_WM_easy']:
#     Input, Output = prepare_WM(i)
#     # print(Input, Output)
# # corresponds 9 batches of 60 sequences, respectively, on normal level
# for i in file_dict['filesList_WM_normal']:
#     Input, Output = prepare_WM(i)
#     # print(Input, Output)
# # corresponds 9 batches of 60 sequences, respectively, on hard level
# for i in file_dict['filesList_WM_hard']:
#     Input, Output = prepare_WM(i)
#     # print(Input, Output)
#
# # WM Anti batches todo: All batches can be treated equally
# # corresponds 9 batches of 60 sequences, respectively, on easy level
# for i in file_dict['filesList_WMAnti_easy']:
#     Input, Output = prepare_WM(i)
#     # print(Input, Output)
# # corresponds 9 batches of 60 sequences, respectively, on normal level
# for i in file_dict['filesList_WMAnti_normal']:
#     Input, Output = prepare_WM(i)
#     # print(Input, Output)
# # corresponds 9 batches of 60 sequences, respectively, on hard level
# for i in file_dict['filesList_WMAnti_hard']:
#     Input, Output = prepare_WM(i)
#     # print(Input, Output)
#
# # WM Ctx1 batches todo: All batches can be treated equally
# # corresponds 9 batches of 60 sequences, respectively, on easy level
# for i in file_dict['filesList_WMCtx1_easy']:
#     Input, Output = prepare_WM(i)
#     # print(Input, Output)
# # corresponds 9 batches of 60 sequences, respectively, on normal level
# for i in file_dict['filesList_WMCtx1_normal']:
#     Input, Output = prepare_WM(i)
#     # print(Input, Output)
# # corresponds 9 batches of 60 sequences, respectively, on hard level
# for i in file_dict['filesList_WMCtx1_hard']:
#     Input, Output = prepare_WM(i)
#     # print(Input, Output)
#
# # WM Ctx2 batches todo: All batches can be treated equally
# # corresponds 9 batches of 60 sequences, respectively, on easy level
# for i in file_dict['filesList_WMCtx2_easy']:
#     Input, Output = prepare_WM(i)
#     # print(Input, Output)
# # corresponds 9 batches of 60 sequences, respectively, on normal level
# for i in file_dict['filesList_WMCtx2_normal']:
#     Input, Output = prepare_WM(i)
#     # print(Input, Output)
# # corresponds 9 batches of 60 sequences, respectively, on hard level
# for i in file_dict['filesList_WMCtx2_hard']:
#     Input, Output = prepare_WM(i)
#     # print(Input, Output)