import torch
import pandas as pd

model0 = torch.load('data/GA/trianed_model_0')
model1 = torch.load('data/GA/trianed_model_1')
model2 = torch.load('data/GA/trianed_model_2')
model3 = torch.load('data/GA/trianed_model_3')
model4 = torch.load('data/GA/trianed_model_4')

stats = pd.read_csv('data/GA/stats.csv')
std = stats.iloc[0]['STD']
mean = stats.iloc[0]['Mean']

