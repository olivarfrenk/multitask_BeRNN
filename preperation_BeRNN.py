import numpy as np
import pandas as pd
import os
# import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Gradient activation function
def get_dist(original_dist):
    '''Get the distance in periodic boundary conditions'''
    return np.minimum(abs(original_dist),2*np.pi-abs(original_dist))
def add_x_loc(x_loc, pref):
    """Input activity given location."""
    dist = get_dist(x_loc - pref)  # periodic boundary
    dist /= np.pi / 8
    return 0.8 * np.exp(-dist ** 2 / 2)


def prepare_DM_error(df): # (model, loss_type, file_location, sequence_on, sequence_off)


    # For bug fixing
    file_location = os.getcwd() + '\\Data CSP\\JW\\7962306_DM_easy_1100.xlsx'
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
        if df_selection['Component Name'][i] == 'Fixation':
            incrementList.append(i + 1)

    # Get average epoch time steps for the selected task in one session
    finalTrialsList = []
    numFixStepsTotal = 0
    numRespStepsTotal = 0
    iterationSteps = 0
    for i in incrementList:
        currentTrial = df_selection[i:i + 2].reset_index().drop(columns=['index']) # need both sequences, as response time comes from difference between first and second
        numFixSteps = round(currentTrial['Onset Time'][0] / 20)  # equal to neuronal time constant of 20ms (Yang, 2019)
        numRespSteps = round(currentTrial['Onset Time'][1] / 20)  # equal to neuronal time constant of 20ms (Yang, 2019)

        currentTrial = df_selection[i:i+1].reset_index().drop(columns=['index']) # need only one sequence
        # Create list with high-sampled rows for both epochs
        currentSequenceList = []
        for j in range(0, numFixSteps+numRespSteps):  # Add time steps
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
    # Delete all rows that are not needed (for either training or testing)
    Input = Input[:, 0:len(incrementList), :]
    Output = Output[:, 0:len(incrementList), :]

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
    for i in range(0, numRespStepsAverage):
        for j in range(0, Output.shape[1]):
            for k in range(2, 34):
                Output[i][j][k] = float(0.05)
    # float all field units of response epoch to 0
    for i in range(numRespStepsAverage, TotalStepsAverage):
        for j in range(0, Output.shape[1]):
            for k in range(2, 34):
                Output[i][j][k] = float(0)
    # float all fixation outputs during response period to 0.05
    for i in range(numRespStepsAverage, TotalStepsAverage):
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
                y_loc[k][j] = np.float(-1)

            if len(nonZerosOutput) == 1:
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

    epochs = {'fix1': (None, numFixStepsAverage),
                'go1': (numFixStepsAverage, None)}

    # Create c_mask
    # c_mask = add_c_mask_BeRNN(Output.shape[0], Output.shape[1], n_output, loss_type, numFixStepsAverage, numRespStepsAverage)

    return Input, Output, y_loc, epochs    #, c_mask