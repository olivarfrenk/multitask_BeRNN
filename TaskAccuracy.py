import os
import pandas as pd

# Participant list
participantList = os.listdir('W:/AG_CSP/Projekte/BeRNN/02_Daten/BeRNN_main/')

particpant = participantList[5] # choose which particpant to analyze
month = '/1/' # choose which month to analyze

percentCorrect_DM, count_DM = 0, 0
percentCorrect_DM_Anti, count_DM_Anti = 0, 0
percentCorrect_EF, count_EF = 0, 0
percentCorrect_EF_Anti, count_EF_Anti = 0, 0
percentCorrect_RP, count_RP = 0, 0
percentCorrect_RP_Anti, count_RP_Anti = 0, 0
percentCorrect_RP_Ctx1, count_RP_Ctx1 = 0, 0
percentCorrect_RP_Ctx2, count_RP_Ctx2 = 0, 0
percentCorrect_WM, count_WM = 0, 0
percentCorrect_WM_Anti, count_WM_Anti = 0, 0
percentCorrect_WM_Ctx1, count_WM_Ctx1 = 0, 0
percentCorrect_WM_Ctx2, count_WM_Ctx2 = 0, 0

# co: Download data as .xlsx long format
list_testParticipant_month = os.listdir('W:/AG_CSP/Projekte/BeRNN/02_Daten/BeRNN_main/' + particpant + month)
for i in list_testParticipant_month:
    # print(i)
    currentFile = pd.read_excel('W:/AG_CSP/Projekte/BeRNN/02_Daten/BeRNN_main/' + particpant + month + i, engine='openpyxl')
    if isinstance(currentFile.iloc[0,28],float) == False: # avoid first rows with state questions .xlsx files
        # print(currentFile.iloc[0,28].split('_trials_')[0])
        if currentFile.iloc[0,28].split('_trials_')[0] == 'DM':
            percentCorrect_DM += currentFile['Store: PercentCorrectDM'][len(currentFile['Store: PercentCorrectDM'])-3]
            count_DM += 1
        if currentFile.iloc[0,28].split('_trials_')[0] == 'DM_Anti':
            percentCorrect_DM_Anti += currentFile['Store: PercentCorrectDMAnti'][len(currentFile['Store: PercentCorrectDMAnti'])-3]
            count_DM_Anti += 1
        if currentFile.iloc[0,28].split('_trials_')[0] == 'EF':
            percentCorrect_EF += currentFile['Store: PercentCorrectEF'][len(currentFile['Store: PercentCorrectEF'])-3]
            count_EF += 1
        if currentFile.iloc[0,28].split('_trials_')[0] == 'EF_Anti':
            percentCorrect_EF_Anti += currentFile['Store: PercentCorrectEF'][len(currentFile['Store: PercentCorrectEF'])-3] # no extra displays for Anti were made
            count_EF_Anti += 1
        if currentFile.iloc[0,28].split('_trials_')[0] == 'RP':
            percentCorrect_RP += currentFile['Store: PercentCorrectRP'][len(currentFile['Store: PercentCorrectRP'])-3]
            count_RP += 1
        if currentFile.iloc[0,28].split('_trials_')[0] == 'RP_Anti':
            percentCorrect_RP_Anti += currentFile['Store: PercentCorrectRPAnti'][len(currentFile['Store: PercentCorrectRPAnti'])-3]
            count_RP_Anti += 1
        if currentFile.iloc[0,28].split('_trials_')[0] == 'RP_Ctx1':
            percentCorrect_RP_Ctx1 += currentFile['Store: PercentCorrectRPCtx1'][len(currentFile['Store: PercentCorrectRPCtx1'])-3]
            count_RP_Ctx1 += 1
        if currentFile.iloc[0,28].split('_trials_')[0] == 'RP_Ctx2':
            percentCorrect_RP_Ctx2 += currentFile['Store: PercentCorrectRPCtx2'][len(currentFile['Store: PercentCorrectRPCtx2'])-3]
            count_RP_Ctx2 += 1
        if currentFile.iloc[0,28].split('_trials_')[0] == 'WM':
            percentCorrect_WM += currentFile['Store: PercentCorrectWM'][len(currentFile['Store: PercentCorrectWM'])-3]
            count_WM += 1
        if currentFile.iloc[0,28].split('_trials_')[0] == 'WM_Anti':
            percentCorrect_WM_Anti += currentFile['Store: PercentCorrectWMAnti'][len(currentFile['Store: PercentCorrectWMAnti'])-3]
            count_WM_Anti += 1
        if currentFile.iloc[0,28].split('_trials_')[0] == 'WM_Ctx1':
            percentCorrect_WM_Ctx1 += currentFile['Store: PercentCorrectWMCtx1'][len(currentFile['Store: PercentCorrectWMCtx1'])-3]
            count_WM_Ctx1 += 1
        if currentFile.iloc[0,28].split('_trials_')[0] == 'WM_Ctx2':
            percentCorrect_WM_Ctx2 += currentFile['Store: PercentCorrectWMCtx2'][len(currentFile['Store: PercentCorrectWMCtx2'])-3]
            count_WM_Ctx2 += 1

acc_DM = percentCorrect_DM/count_DM
acc_DM_Anti = percentCorrect_DM_Anti/count_DM_Anti
acc_EF = percentCorrect_EF/count_EF
acc_EF_Anti = percentCorrect_EF_Anti/count_EF_Anti
acc_WM = percentCorrect_WM/count_WM
acc_WM_Anti = percentCorrect_WM_Anti/count_WM_Anti
acc_WM_Ctx1 = percentCorrect_WM_Ctx1/count_WM_Ctx1
acc_WM_Ctx2 = percentCorrect_WM_Ctx2/count_WM_Ctx2
acc_RP = percentCorrect_RP/count_RP
acc_RP_Anti = percentCorrect_RP_Anti/count_RP_Anti
acc_RP_Ctx1 = percentCorrect_RP_Ctx1/count_RP_Ctx1
acc_RP_Ctx2 = percentCorrect_RP_Ctx2/count_RP_Ctx2


# pd.DataFrame(data={'acc_WM_Ctx2':[acc_WM_Ctx2]})




