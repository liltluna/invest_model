# This is a new version of phase1 constructed with pandas_ta
import os
import pathlib
import pandas as pd
import pandas_ta as ta
from formula import *
from sklearn.preprocessing import MinMaxScaler
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

start = 6
end = 21
theta = 0.3
t_forward = 7

column_names = ["date", "open", "high", "low",
                "close", "preClose", "vol"]  # Define column names

file_path = pathlib.Path('./dataset/reversed_all_data.csv')
dir_path = file_path.parent
source_data = pd.read_csv(file_path, header=None, names=column_names)
source_data['date'] = pd.to_datetime(
    source_data["date"], format="%Y-%m-%d")
source_data.set_index('date', inplace=True)

LABELS_list = calculate_trichotomous_LABELS(
    source_data['close'].tolist(), t_forward=t_forward, theta=theta)
source_data.drop(source_data.head(t_forward).index, inplace=True)
source_data['LABELS'] = LABELS_list
source_data.to_csv(dir_path/'ration_label/ration.csv')
