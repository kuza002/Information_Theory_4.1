import torch
from openpyxl import load_workbook
import pandas as pd
import numpy as np

data_set_length = 887

#############################################################################
#                                  Load Dataset                             #
#############################################################################

# Get workbook
wb = load_workbook('БД Титаник.xlsx')

# Get sheet
sheet = wb.active

# Get targets values and input parametors

data = np.zeros(shape=(data_set_length, 7))

for i, cell_obj in enumerate(sheet["A4":"G890"]):
    for j, target in enumerate(cell_obj):
        data[i][j] = float(target.value)

# Split data set into training, validation and test set

training_set_length = int(data_set_length*0.7)
validation_set_length = int(data_set_length*0.15)
test_set_length = int(data_set_length*0.15)

pd.DataFrame(data
             [:training_set_length]).to_csv('dataset/training_set.csv', index=False)
pd.DataFrame(data
             [training_set_length:training_set_length+validation_set_length]).to_csv('dataset/validation_set.csv', index=False)
pd.DataFrame(data
             [training_set_length+validation_set_length:training_set_length+validation_set_length+test_set_length]).to_csv('dataset/test_set.csv', index=False)



