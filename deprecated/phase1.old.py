import os
import pandas as pd
from utils.formula import *
from sklearn.preprocessing import MinMaxScaler
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

start = 6
end = 21
data_name_list = ['RSI', 'WILLIANMS', 'SMA', 'EMA', 'WMA', 'HMA',
                  'TEMA', 'CCI', 'CMO', 'MACD', 'PPO', 'ROC', 'CMF', 'DMI', 'SAR']

# Define column names
column_names = ["date", "open", "high", "low", "close", "preClose", "vol"]
# Read CSV file into a DataFrame

for i in range(0, 10):
    file_path_values = ['./dataset/set-{}/reversed_test_data.csv'.format(i), 
                        './dataset/set-{}/reversed_train_data.csv'.format(i)]
    dir = './dataset/set-{}/'.format(i)
    for file_path in file_path_values:

        data = pd.DataFrame()
        df = pd.read_csv(file_path, header=None, names=column_names)
        df['date'] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        # Convert DataFrame to list of dictionaries

        low_values = df['low'].tolist()
        vol_values = df['vol'].tolist()
        open_values = df['open'].tolist()
        high_values = df['high'].tolist()
        close_values = df['close'].tolist()


        for period in range(start, end):
            # those with period
            RSI = calculate_RSI(close_values, period)
            WILLIAMS = calculate_WILLIAMS_PERSENT_R(
                high_values, low_values, close_values, period)
            SMA = calculate_SMA(close_values, period)
            EMA = calculate_EMA(close_values, period)
            WMA = calculate_WMA(close_values, period)
            HMA = calculate_smooth_HMA(close_values, period)

            CCI = calculate_CCI(high_values, low_values, close_values, period)
            CMO = calculate_CMO(close_values, period)
            ROC = calculate_ROC(close_values, period)
            DMI = calculate_ADX(high_values, low_values, close_values, period)

            # those without a period
            TEMA = calculate_triple_EMA(close_values)
            MACD = calculate_MACD(close_values)
            PPO = calculate_PPO(close_values)
            CMF = calculate_CMF(close_values, high_values, low_values, vol_values)
            SAR = calculate_SAR(high_values, low_values)

            data['RSI-{}'.format(period)] = RSI
            data['WILLIANMS-{}'.format(period)] = WILLIAMS
            data['SMA-{}'.format(period)] = SMA
            data['EMA-{}'.format(period)] = EMA
            data['WMA-{}'.format(period)] = WMA
            data['HMA-{}'.format(period)] = HMA
            data['TEMA-{}'.format(period)] = TEMA
            data['CCI-{}'.format(period)] = CCI
            data['CMO-{}'.format(period)] = CMO
            data['MACD-{}'.format(period)] = MACD
            data['PPO-{}'.format(period)] = PPO
            data['ROC-{}'.format(period)] = ROC
            data['CMF-{}'.format(period)] = CMF
            data['DMI-{}'.format(period)] = DMI
            data['SAR-{}'.format(period)] = SAR

        # calculate labels
        LABELS = calculate_LABELS(close_values)


        # set the data order
        order = ['LABELS', 'PRICE']
        for index in data_name_list:
            for i in range(start, end):
                order.append('{}-{}'.format(index, i))

        # 将float('nan')替换为NaN
        data.replace({float('nan'): np.nan}, inplace=True)
        data['LABELS'] = LABELS
        data['PRICE'] = close_values
        data = data.dropna()

        verbose_info = pd.DataFrame(data, columns= ['LABELS', 'PRICE'])
        verbose_info = verbose_info.reset_index(drop=True)
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(data)  

        scaled_data_df = pd.DataFrame(scaled_data, columns=data.columns)
        scaled_data_df['LABELS'] = verbose_info['LABELS']
        scaled_data_df['PRICE'] = verbose_info['PRICE']
        scaled_data_df = scaled_data_df[order]

        new_file_path = dir + 'output_phase2_{}.csv'.format(file_path.split('/')[-1].split('_')[-2])
        print('{}-saved...'.format(new_file_path))
        scaled_data_df.round(2).to_csv(new_file_path, index=False, header=None)

