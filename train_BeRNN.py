# Create new gathered .xlsx group files

import os
import pandas as pd

file_path = 'W:/AG_CSP/Projekte/BeRNN/02_Daten/BeRNN_main/BeRNN_01/1/data_exp_149474-v2_task-bx2n-9621849.xlsx'


def split_name(name):
    elements = name.split('_')

    if len(elements) == 4:
        return elements[0]  # Return the first element for a 4-element name
    elif len(elements) == 5:
        return elements[0] + '_' + elements[1]  # Return the first two elements for a 5-element name
    else:
        # Handle other cases if needed
        return elements


# Function to process a single Excel file
def process_excel_file(file_path):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_path, engine='openpyxl')

    # Check the cell value to determine the group type
    group_type = split_name(df.at[0, 'Spreadsheet'])  # Replace 'GroupType' with the actual column name

    # Process the data and extract sequences based on the group type
    processed_data = process_data(df, group_type)  # Implement your custom processing logic

    return group_type, processed_data


# Function to process the data and extract sequences
def process_data(df, group_type):

    if group_type.split('_')[0] == 'DM':
        Input, Output, y_loc, epochs = prepare_DM_error(df)
    elif group_type.split('_')[0] == 'EF':
        Input, Output, y_loc, epochs = prepare_EF_error(df)
    elif group_type.split('_')[0] == 'RP':
        Input, Output, y_loc, epochs = prepare_RP_error(df)
    elif group_type.split('_')[0] == 'WM':
        Input, Output, y_loc, epochs = prepare_WM_error(df)

    return processed_data