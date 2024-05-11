import tushare as ts
import pandas as pd
import shutil
import os


start_year = 2009
gap_year = 9  

ts_code = '000700.SZ'

pro = pro = ts.pro_api(
    '19c1ab37c30ca784d8658cf1050c0aba8eef6cb8d7cda5d0744d6bfb')

if not os.path.exists('dataset'):
    os.mkdir('dataset')
else:
    pass

# keep all the data in a file
columns_to_drop = ['ts_code', 'change', 'pct_chg', 'amount']  # 删除指定的列
df = pro.daily(ts_code=ts_code, start_date='20120401', end_date='20240401')

print(f'GET: Length of dataset: {len(df)}')

df.drop(columns=columns_to_drop, inplace=True)
df['trade_date'] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
df_reversed = df[::-1]
df_reversed.to_csv(f'./dataset/reversed_all_data_{ts_code}.csv',
                   index=False, header=False)  # index=False 防止保存索引列



# if you want to segment the dataset, using the following code ...

# for i in range(4):  
#     dir = './dataset/set-{}'.format(i)
#     os.mkdir(dir)
#     start_date = '{}-01-01'.format(start_year + i)
#     end_date = '{}-01-01'.format(start_year + i + gap_year)
#     test_start_date = end_date
#     test_end_date = '{}-01-01'.format(start_year + i + gap_year + 1)

#     selected_data = df[(df['trade_date'] >= pd.to_datetime(start_date)) & (
#         df['trade_date'] <= pd.to_datetime(end_date))]
#     reversed_selected_data = selected_data[::-1]

#     test_selected_data = df[(df['trade_date'] >= pd.to_datetime(test_start_date)) & (
#         df['trade_date'] <= pd.to_datetime(test_end_date))]
#     test_reversed_selected_data = test_selected_data[::-1]

#     reversed_selected_data.to_csv(
#         dir + '/reversed_train_data.csv', index=False, header=False)
#     test_reversed_selected_data.to_csv(
#         dir + '/reversed_test_data.csv', index=False, header=False)  # index=False 防止保存索引列
#     print('sd:{}, ed:{}, tsd:{}, ted:{}'.format(
#         start_date, end_date, test_start_date, test_end_date))
#     print('save {}...'.format(dir))
