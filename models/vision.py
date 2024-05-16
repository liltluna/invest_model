import pandas as pd
import matplotlib.pyplot as plt
from models.dataset import *
from models.models import *
from pathlib import Path


def plot_loss_result(config):

    window = 10
    statistic_dir = Path(
        f"./{config['result_folder']}/{config['ts_code']}/{config['model_name']}")
    statistic_path = statistic_dir/"statistic.csv"
    # window = 10
    # 步骤1: 加载CSV文件中的数据
    data = pd.read_csv(statistic_path)

    # 计算损失和准确率的移动平均值
    data['Smoothed Loss'] = data['loss'].rolling(window=window).mean()
    data['Smoothed Accuracy'] = data['acc'].rolling(
        window=window).mean()  # 假设'accuracy'是准确率的列名

    # 对每个AUC相关的列计算移动平均值
    for auc_col in ['hold', 'sell', 'buy']:
        data[f'Smoothed {auc_col}'] = data[auc_col].rolling(
            window=window).mean()
    # 提取epoch、平滑后的loss和accuracy数据

    epochs = data['epoch']
    losses = data['Smoothed Loss']
    accuracies = data['Smoothed Accuracy']

    # 步骤3: 绘制loss和accuracy曲线
    plt.figure(figsize=(10, 5))  # 设置图形大小
    plt.ylim(0, 1)
    # 绘制loss曲线
    plt.plot(epochs, losses, label='Loss', color='blue')  # 绘制平滑后的loss曲线，颜色设为蓝色

    # 绘制accuracy曲线
    plt.plot(epochs, accuracies, label='Accuracy', color='red')
    # 绘制新的AUC曲线
    plt.plot(epochs, data['Smoothed hold'], label='Hold AUC', color='orange')
    plt.plot(epochs, data['Smoothed sell'], label='Sell AUC', color='purple')
    plt.plot(epochs, data['Smoothed buy'], label='Buy AUC', color='brown')

    # 添加标题和坐标轴标签
    plt.title(f"{config['model_name']} On {config['ts_code']}")
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(False)  # 原本为False，这里假设你想显示网格，根据需要可以调整

    # 直接保存图像，而不是显示
    plt.savefig(statistic_dir /
                f"{config['model_name']}_On_{config['ts_code']}.png")


def calculate_Captial(total_money, current_stock, close_price):
    return total_money + (current_stock * close_price)


def plot_finicial_evalutaion_comparation(config):
    # define column names
    column_names = ["date", "open", "high", "low", "close", "preClose", "vol"]

    # define data path
    dataset_path = Path(f"./dataset/reversed_all_data_{config['ts_code']}.csv")
    result_dir = Path(f"./{config['result_folder']}/{config['ts_code']}")

    # read the price infomation
    price_df = pd.read_csv(dataset_path, header=None, names=column_names)
    price_df['date'] = pd.to_datetime(price_df['date'])
    price_df.set_index('date', inplace=True)  # 设置日期为索引以便于数据对齐

    results = ['CNN/epoch/epoch-199.csv',
               'MLP/epoch/epoch-199.csv',
               'TECEC_1/epoch/epoch-199.csv',
               'TECEC_2/epoch/epoch-199.csv',
               'RNN/epoch/epoch-145.csv',
               'LSTM/epoch/epoch-166.csv']

    # 初始化一个字典来存放所有模型的预测结果
    predictions = {}

    # 读取并处理每个结果文件
    for result_file in results:
        # 读取CSV
        result_epoch_path = result_dir/result_file
        result_df = pd.read_csv(result_epoch_path)
        result_df['Date'] = pd.to_datetime(result_df['Date'])  # 确保日期格式正确

        # 使用文件路径的第一部分作为列名
        model_name = result_file.split('/')[0]
        result_df.rename(columns={'PREDICTED': model_name}, inplace=True)

        # 丢弃不需要的列（如'TARGET'），如果存在的话
        if 'TARGET' in result_df.columns:
            result_df.drop(columns=['TARGET'], inplace=True)

        # 合并到预测字典中，以日期为键
        for date, pred in result_df.set_index('Date').iterrows():
            if date not in predictions:
                predictions[date] = {}
            predictions[date][model_name] = pred[model_name]

    # 将预测转换为DataFrame并与价格数据合并
    prediction_df = pd.DataFrame(predictions).T.fillna(0)  # 转置并用0填充缺失值
    merged_df = price_df.join(prediction_df, how='inner')  # 根据索引（日期）与价格数据合并
    # 注意：这里使用了join方法，因为预测DataFrame的索引已经是日期，且我们希望保留所有价格数据的日期行
    targets = [x.split('/')[0] for x in results]

    result_financial_path = result_dir/'financial_comparation_result.txt'

    for asset in targets:

        # 初始化指标变量
        total_money = 10000.0
        total_stock = 0
        success_transaction_count = 0
        total_percent_profit = 0
        total_transaction_length = 0
        current_stock = 0
        transaction_count = 0
        captial_column_name = f"{asset}-On-{config['ts_code']}-CAPTIAL"

        merged_df[captial_column_name] = total_money

        # 定义一些临时变量
        last_operation_type = None  # 用于追踪上一次的操作类型
        transaction_lengths = []  # 存储每次交易的持续天数
        idle_days = 0  # 累计空闲天数
        last_sell_day = None
        for i, row in merged_df.iterrows():

            # 根据操作更新资金和股票状态
            if row[asset] == 1:  # BUY
                buy_price = row['open']
                shares_bought = total_money / buy_price
                current_stock += shares_bought
                transaction_count += 1
                total_money = total_money - shares_bought * buy_price

            elif row[asset] == 2:  # SELL
                sell_price = row['close']
                # profit = (sell_price - buy_price) * shares_bought
                total_money = total_money + sell_price * current_stock
                current_stock = 0
                transaction_count += 1
                success_transaction_count += 1
                # total_percent_profit += profit
                # total_transaction_length += (i - last_sell_day) if last_sell_day is not None else 0
                last_sell_day = i
            elif row[asset] == 0:  # HOLD
                pass  # 或者累加闲置时间等其他操作

            # 更新每日结束资金量（考虑到HOLD的情况，资金无变化，如果是BUY或SELL已在上面处理）
            # if row[target] != 1:  # 不是BUY的那一天资金不变
            #     merged_df.at[i, 'daily_end_money'] = merged_df.at[i, 'daily_start_money']
            # else:  # 如果是BUY，资金已经通过买卖操作更新，直接记录total_money
            #     merged_df.at[i, 'daily_end_money'] = total_money

            # 计算并追踪交易长度和空闲时间
            if row[asset] in [1, 2]:  # 对于'BUY'或'SELL'操作
                if last_operation_type != row[asset]:  # 如果这是新的买卖操作序列的开始
                    if last_operation_type == 0:  # 如果上一个操作是'HOLD'
                        idle_days += 1  # 增加空闲天数
                    if isinstance(last_sell_day, pd.Timestamp) and isinstance(i, pd.Timestamp):
                        days_passed = (i - last_sell_day).days
                        transaction_lengths.append(days_passed)
                    else:
                        transaction_lengths.append(0)
                last_operation_type = row[asset]
            elif row[asset] == 0:  # HOLD
                if last_operation_type in [1, 2]:  # 如果前一个操作是交易操作
                    idle_days += 1  # 增加空闲天数

            merged_df.at[i, captial_column_name] = calculate_Captial(
                total_money, current_stock, row['close'])

        # 删除第一天的'daily_start_money'，因为它是初始资金，没有前一日的记录
        # if not merged_df.empty:
        #     merged_df.drop(columns=['daily_start_money'], inplace=True, errors='ignore')
        # 循环结束后，计算各项指标
        # 年化收益率 (AR) 需要总年数作为输入，这里假设数据覆盖了1年，可根据实际情况调整
        numberOfYears = 1
        totalMoney = total_money + \
            (current_stock * merged_df.iloc[-1]['close'])
        startMoney = 10000.0
        AR = (((totalMoney / startMoney) ** (1/numberOfYears)) - 1) * 100

        # 成功交易比例 (PoS)
        PoS = (success_transaction_count / transaction_count) * \
            100 if transaction_count > 0 else 0

        # 平均交易时间 (AnT) 和平均交易长度 (L) 需要先计算总交易长度
        totalTransLength = sum(transaction_lengths)
        AnT = transaction_count / numberOfYears if transaction_count > 0 else 0
        L = totalTransLength / transaction_count if transaction_count > 0 else 0

        # 平均百分比利润 (ApT) 需要总百分比利润，这里简化处理，实际应用中应基于每次交易的盈利计算
        # ApT 目前没有直接计算，需要具体每次交易的利润来累计totalPercentProfit

        # 空闲率 (IdleR)
        IdleR = (merged_df.shape[0] - idle_days -
                 totalTransLength) / merged_df.shape[0] * 100

        # 输出或使用这些指标
        print(f"{asset}:")
        print(f"年化收益率 (AR): {AR:.2f}%")
        print(f"成功交易比例 (PoS): {PoS:.2f}%")
        print(f"平均交易时间 (AnT): {AnT:.2f}")
        print(f"平均交易长度 (L): {L:.2f}")
        print(f"空闲率 (IdleR): {IdleR:.2f}%")
        print("-" * 30)
        # 注意：这里的ApT计算没有直接实现，因为涉及到对每次交易利润的具体计算，这需要在买卖操作中累积。

        with open(result_financial_path, "a") as file:
            file.write(f"{asset}:\n")
            file.write(f"年化收益率 (AR): {AR:.2f}%\n")
            file.write(f"成功交易比例 (PoS): {PoS:.2f}%\n")
            file.write(f"平均交易时间 (AnT): {AnT:.2f}\n")
            file.write(f"平均交易长度 (L): {L:.2f}\n")
            file.write(f"空闲率 (IdleR): {IdleR:.2f}%\n")
            file.write("-" * 30 + "\n")

    df = merged_df
    # 设置统一的线条宽度
    line_width = 0.7  # 你可以根据需要调整这个值

    plt.figure(figsize=(14, 7))

    df[f"MLP-On-{config['ts_code']}-CAPTIAL"].plot(
        label=f"MLP-On-{config['ts_code']}-CAPTIAL", marker='o', linewidth=line_width)
    df[f"CNN-On-{config['ts_code']}-CAPTIAL"].plot(
        label=f"CNN-On-{config['ts_code']}-CAPTIAL", marker='s', linewidth=line_width)
    df[f"TECEC_1-On-{config['ts_code']}-CAPTIAL"].plot(
        label=f"TECEC_1-On-{config['ts_code']}-CAPTIAL", marker='^', linewidth=line_width)
    df[f"TECEC_2-On-{config['ts_code']}-CAPTIAL"].plot(
        label=f"TECEC_2-On-{config['ts_code']}-CAPTIAL", marker='x', linewidth=line_width)
    df[f"RNN-On-{config['ts_code']}-CAPTIAL"].plot(
        label=f"RNN-On-{config['ts_code']}-CAPTIAL", marker='>', linewidth=line_width)
    df[f"LSTM-On-{config['ts_code']}-CAPTIAL"].plot(
        label=f"LSTM-On-{config['ts_code']}-CAPTIAL", marker='v', linewidth=line_width)

    plt.title(f"Capital Evaluation On {config['ts_code']}")
    plt.xlabel('Date')
    plt.ylabel('Capital Value')
    plt.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(result_dir/f"Capital_Evaluation_on_{config['ts_code']}.png")

    # plot the price of stock
    plt.figure(figsize=(14, 7))  # 设置图形的尺寸

    # 绘制close价格随日期变化的折线图
    plt.plot(merged_df.index, merged_df['close'],
             label='Close Price', color='blue')
    # 设置图形标题和坐标轴标签
    plt.title(f"Close Price of {config['ts_code']}")
    plt.xlabel('Date')
    plt.ylabel('Close Price')

    # 添加网格线
    plt.grid(False)
    # 添加图例
    plt.legend()
    # 优化x轴日期显示，避免重叠
    plt.gcf().autofmt_xdate()  # 自动旋转并调整x轴日期标签的间距
    # 显示图形
    plt.savefig(result_dir/f"Close_Price_of_{config['ts_code']}.png")
