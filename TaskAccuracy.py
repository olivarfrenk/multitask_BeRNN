import os
import pandas as pd

# Participant list
participantList = os.listdir(os.getcwd() + '/BeRNN_data/')

particpant = participantList[5]
month = '/1/'

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

list_testParticipant_month1 = os.listdir(os.getcwd() + '/BeRNN_data/' + participantList[5] + month)
for i in list_testParticipant_month1:
    currentFile = pd.read_excel(os.getcwd() + '/BeRNN_data/' + participantList[5] + month + i, engine='openpyxl')
    print(currentFile)
    if currentFile['Spreadsheet'][0].split('_trials_')[0] == 'WM_Ctx2':
        percentCorrect_WM_Ctx2 += currentFile['Store: PercentCorrectWMCtx2'][len(currentFile['Store: PercentCorrectWMCtx2'])-3]
        count_WM_Ctx2 += 1








