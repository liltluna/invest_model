# This is a new version of phase1 constructed with pandas_ta
import os
import pathlib
import pandas as pd
import pandas_ta as ta
from utils.formula import *
from sklearn.preprocessing import MinMaxScaler
from warnings import simplefilter
from pathlib import Path
from models.config import CONFIG

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# config 
start = 6
end = 21
ts_code = CONFIG['ts_code']
# label methods: graph_label or ...
label_method = 'graph_label'
precision = 6
# train and test split
split_date = pd.Timestamp('2023-01-01')

data_name_list = ['rsi', 'willr', 'sma', 'ema', 'wma', 'hma',
                  'tema', 'cci', 'cmo', 'macd_h', 'ppo_h', 'roc', 'cmf', 'adx', 'psar']
# Define column names
column_names = ["date", "open", "high", "low",
                "close", "preClose", "vol"]  
# Read CSV file into a DataFrame
file_path = pathlib.Path(f'./dataset/reversed_all_data_{ts_code}.csv')
dir_path = file_path.parent
Path(dir_path/label_method).mkdir(parents=True, exist_ok=True)

temp_data = pd.DataFrame()
temp_frames = []

source_data = pd.read_csv(file_path, header=None, names=column_names)
# source_data = source_data.drop(index=0)
source_data['date'] = pd.to_datetime(
    source_data["date"], format="%Y-%m-%d")
source_data.set_index('date', inplace=True)

LABELS_list = calculate_LABELS(source_data['close'].tolist())
temp_data.index = source_data.index.copy()
temp_data['LABELS'] = LABELS_list
temp_data.replace({
    float('nan'): np.nan}, inplace=True)
temp_data['PRICE'] = source_data['close']
temp_data.replace({float('nan'): np.nan}, inplace=True)
source_data = source_data.dropna()

for length in range(start, end):
    rsi = ta.rsi(source_data['close'], length)
    willr = ta.willr(
        source_data['high'], source_data['low'], source_data['close'], length)
    ema = ta.ema(source_data['close'], length)
    sma = ta.sma(source_data['close'], length)
    wma = ta.fwma(source_data['close'], length)
    hma = ta.hma(source_data['close'], length)
    tema = ta.tema(source_data['close'], length)
    cci = ta.cci(source_data['high'], source_data['low'],
                source_data['close'], length)
    cmo = ta.cmo(source_data['close'], length)
    macd_h = ta.macd(source_data['close'], offset=length).iloc[:, 1]
    ppo_h = ta.ppo(source_data['close'], offset=length).iloc[:, 1]
    roc = ta.roc(source_data['close'], length)
    cmf = ta.cmf(open_=source_data['open'], high=source_data['high'], low=source_data['low'],
                close=source_data['close'], volume=source_data['vol'], length=6)
    adx = ta.adx(source_data['high'], source_data['low'],
                source_data['close'], length).iloc[:, 0]
    psar = ta.psar(high=source_data['high'], low=source_data['low'],
                close=source_data['close'], offset=length).iloc[:, 2]

    temp_frame = pd.DataFrame({
        'rsi_{}'.format(length): rsi,
        'willr_{}'.format(length): willr,
        'ema_{}'.format(length): ema,
        'sma_{}'.format(length): sma,
        'wma_{}'.format(length): wma,
        'hma_{}'.format(length): hma,
        'tema_{}'.format(length): tema,
        'cci_{}'.format(length): cci,
        'cmo_{}'.format(length): cmo,
        'macd_h_{}'.format(length): macd_h,
        'ppo_h_{}'.format(length): ppo_h,
        'roc_{}'.format(length): roc,
        'cmf_{}'.format(length): cmf,
        'adx_{}'.format(length): adx,
        'psar_{}'.format(length): psar,
    })

    temp_frames.append(temp_frame)

temp_frames.append(temp_data)
# Concatenate all temporary frames along the columns axis
data = pd.concat(temp_frames, axis=1)

data = data.dropna()
order = ['LABELS', 'PRICE']
for index in data_name_list:
    for i in range(start, end):
        order.append('{}_{}'.format(index, i))

verbose_info = pd.DataFrame(data, columns=['LABELS', 'PRICE'])
verbose_info = verbose_info.reset_index(drop=True)

data = data[~np.isinf(data).any(axis=1)]  # 删除含有无穷大的行

scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(data)

scaled_data_df = pd.DataFrame(scaled_data, columns=data.columns)
scaled_data_df['LABELS'] = verbose_info['LABELS']
scaled_data_df['PRICE'] = verbose_info['PRICE']
scaled_data_df = scaled_data_df[order]
scaled_data_df.index = data.index

result_train_file_path = dir_path/f'{label_method}/{ts_code}-train.csv'
result_test_file_path = dir_path/f'{label_method}/{ts_code}-test.csv'

train_df = scaled_data_df[scaled_data_df.index < split_date]
test_df = scaled_data_df[scaled_data_df.index >= split_date]

train_df.round(precision).to_csv(result_train_file_path, index=True, header=None)
print('{}-saved...'.format(result_train_file_path))

test_df.round(precision).to_csv(result_test_file_path, index=True, header=None)
print('{}-saved...'.format(result_test_file_path))
