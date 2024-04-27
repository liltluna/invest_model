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
data_name_list = ['rsi', 'willr', 'sma', 'ema', 'wma', 'hma',
                  'tema', 'cci', 'cmo', 'macd_h', 'ppo_h', 'roc', 'cmf', 'adx', 'psar']
column_names = ["date", "open", "high", "low",
                "close", "preClose", "vol"]  # Define column names

file_path = pathlib.Path('./dataset/reversed_all_data.csv')
dir_path = file_path.parent
temp_data = pd.DataFrame()
temp_frames = []

source_data = pd.read_csv(file_path, header=None, names=column_names)
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

scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(data)

scaled_data_df = pd.DataFrame(scaled_data, columns=data.columns)
scaled_data_df['LABELS'] = verbose_info['LABELS']
scaled_data_df['PRICE'] = verbose_info['PRICE']
scaled_data_df = scaled_data_df[order]

new_file_path = dir_path / 'output_phase2_all_data.csv'
if os.path.exists(new_file_path):
    os.remove(new_file_path)
scaled_data_df.round(2).to_csv(new_file_path, index=False, header=None)
print('{}-saved...'.format(new_file_path))
