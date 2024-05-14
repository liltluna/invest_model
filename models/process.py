import torch
import csv
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from models.dataset import *
from models.models import *
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer



# Define the device
DEVICE = "cuda:0" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
# device = "cpu"
print("Using device:", DEVICE)
if (DEVICE == 'cuda:0'):
    print(f"Device name: {torch.cuda.get_device_name(DEVICE.index)}")
    print(
        f"Device memory: {torch.cuda.get_device_properties(DEVICE.index).total_memory / 1024 ** 3} GB")
else:
    print("NOTE: If you have a GPU, consider using it for training.")
DEVICE = torch.device(DEVICE)


# get dataloader
def get_ds(config):
    train_path = f"./{config['data_folder']}/{config['label_method']}/{config['ts_code']}-train.csv"
    test_path = f"./{config['data_folder']}/{config['label_method']}/{config['ts_code']}-test.csv"

    ds_train = BasicGraphDataset(train_path, 'train')
    ds_test = BasicGraphDataset(test_path, 'test')

    train_ds = GraphDataset(
        ds=ds_train, seq_len=config['seq_len'], num_classes=config["num_classes"])
    val_ds = GraphDataset(
        ds=ds_test, seq_len=config['seq_len'], num_classes=config["num_classes"])

    train_dataloader = DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader


def run_validation(model, validation_ds, config, device, print_msg, epoch):
    model.eval()
    output_count = 0
    match_count = 0
    
    # define indicators to save
    fieldnames = ['Date', 'TARGET', 'PREDICTED']

    # define the file path
    result_dir = Path(
        f"./{config['result_folder']}/{config['ts_code']}/{config['model_name']}")
    result_dir.mkdir(parents=True, exist_ok=True)
    epoch_path = result_dir/f'epoch/epoch-{epoch}.csv'

    with open(epoch_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # 初始化AUC列表，用于存储整个验证集上每个样本属于每个类别的预测概率
    all_probs = [[], [], []]
    true_labels = []
    # Initialize a list to store AUC scores for each class
    aucs_per_class = []

    with torch.no_grad():
        for batch in validation_ds:
            target = int(batch['label'][0].item())
            date = batch['date'][0]
            output_count += 1
            nn_input = batch["nn_input"].to(device)  # (b, seq_len)
            # check that the batch size is 1
            assert nn_input.size(0) == 1, "Batch size must be 1 for validation"
            # Run the tensors through the encoder, decoder and the projection layer
            nn_output = model.forward(nn_input)

            # 确保nn_output是概率形式，如果是logits，则使用softmax转换
            probabilities = torch.softmax(
                nn_output, dim=1).squeeze().cpu().numpy()

            # count the matched result
            _, predicted_labels = torch.max(nn_output, dim=1)
            if target == predicted_labels:
                match_count += 1

            # 收集每个样本的预测概率
            for i, prob in enumerate(probabilities):
                all_probs[i].append(prob)  # 只保留对应类别的概率
            true_labels.append(target)
            # out put the sample
            if output_count < 3:
                # Print the source, target and model output
                print_msg('-'*30)
                # print_msg(f"{f'SOURCE: ':>12.30}{source_img} ...")
                print_msg(f"{f'Date: ':>12}{date}")
                print_msg(f"{f'TARGET: ':>12}{target}")
                print_msg(f"{f'PREDICTED: ':>12}{predicted_labels}")
                # print_msg(f"{f'TYPE: ':>12}{predict}")

            # 再次打开文件并追加写入数据行
            with open(epoch_path, 'a', newline='', encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                # 将数据写入
                writer.writerow({'Date': date, 'TARGET': int(
                    target), 'PREDICTED': int(predicted_labels)})

        accuracy = match_count / len(validation_ds)
        print(f'accuracy: {accuracy}')

    # calculate the auc
    lb = LabelBinarizer()
    lb.fit(range(max(true_labels) + 1))  # Fit on the range of y_true
    y_true_bin = lb.transform(true_labels)
    for i in range(lb.classes_.size):
        # Extract true labels and predicted probabilities for the current class vs rest
        y_true_class = y_true_bin[:, i]
        y_pred_class_proba = all_probs[i]
        
        # Compute ROC AUC for the current class
        auc = roc_auc_score(y_true_class, y_pred_class_proba)
        aucs_per_class.append(auc)

    return {"accuracy": accuracy,
            "class0": aucs_per_class[0],
            "class1": aucs_per_class[1],
            "class2": aucs_per_class[2]}


def train(model, device, config):
    result_dir = Path(
        f"./{config['result_folder']}/{config['ts_code']}/{config['model_name']}")
    epoch_dir = result_dir/'epoch'
    epoch_dir.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ['epoch', 'loss', 'acc', 'hold', 'sell', 'buy']

    # Make sure the weights folder exists
    # Path(f"./{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader = get_ds(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0

    loss_fn = nn.CrossEntropyLoss().to(device)

    with open(result_dir/"statistic.csv", 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(
            train_dataloader, desc=f"Processing Epoch {epoch:03d}")
        for batch in batch_iterator:

            nn_input = batch['nn_input'].to(device)  # (b, seq_len)
            # Run the tensors through the encoder, decoder and the projection layer
            nn_output = model.forward(nn_input)

            # Compare the output with the label
            # label = batch['label'].to(device)
            # 调整标签
            label = batch['label'].long().to(device)

            # 使用NLLLoss
            # Compute the loss using a simple cross entropy
            loss = loss_fn(nn_output, label)
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        val_result = run_validation(model=model, validation_ds=val_dataloader, device=device,
                                    print_msg=lambda msg: batch_iterator.write(msg), epoch=epoch, config=config)

        with open(result_dir/"statistic.csv", 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # 将数据写入
            writer.writerow(
                {'epoch': epoch, 'loss': loss.item(), 'acc': val_result['accuracy'], 'hold': val_result['class0'], 'sell': val_result['class1'], 'buy': val_result['class2']})


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
    data['Smoothed Accuracy'] = data['acc'].rolling(window=window).mean()  # 假设'accuracy'是准确率的列名

    # 对每个AUC相关的列计算移动平均值
    for auc_col in ['hold', 'sell', 'buy']:
        data[f'Smoothed {auc_col}'] = data[auc_col].rolling(window=window).mean()
    # 提取epoch、平滑后的loss和accuracy数据

    epochs = data['epoch']
    losses = data['Smoothed Loss']
    accuracies = data['Smoothed Accuracy']

    # 步骤3: 绘制loss和accuracy曲线
    plt.figure(figsize=(10, 5))  # 设置图形大小

    # 绘制loss曲线
    plt.plot(epochs, losses, label='Loss', color='blue')  # 绘制平滑后的loss曲线，颜色设为蓝色

    # 绘制accuracy曲线
    plt.plot(epochs, accuracies, label='Accuracy',color='red') 
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
    column_names = ["date", "open", "high", "low","close", "preClose", "vol"]  
    
    # define data path
    dataset_path = Path(f"./dataset/reversed_all_data_{config['ts_code']}.csv")
    result_dir = Path(f"./{config['result_folder']}/{config['ts_code']}")

    # read the price infomation
    price_df = pd.read_csv(dataset_path, header=None, names = column_names)
    price_df['date'] = pd.to_datetime(price_df['date'])
    price_df.set_index('date', inplace=True)  # 设置日期为索引以便于数据对齐

    results = ['CNN/epoch/epoch-166.csv', 
            'MLP/epoch/epoch-25.csv',
            'TECEC_1/epoch/epoch-195.csv', 
            'TECEC_2/epoch/epoch-196.csv']

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


            merged_df.at[i, captial_column_name] = calculate_Captial(total_money, current_stock, row['close'])

            
        # 删除第一天的'daily_start_money'，因为它是初始资金，没有前一日的记录
        # if not merged_df.empty:
        #     merged_df.drop(columns=['daily_start_money'], inplace=True, errors='ignore')
        # 循环结束后，计算各项指标
        # 年化收益率 (AR) 需要总年数作为输入，这里假设数据覆盖了1年，可根据实际情况调整
        numberOfYears = 1
        totalMoney = total_money + (current_stock * merged_df.iloc[-1]['close'])
        startMoney = 10000.0
        AR = (((totalMoney / startMoney) ** (1/numberOfYears)) - 1) * 100

        # 成功交易比例 (PoS)
        PoS = (success_transaction_count / transaction_count) * 100 if transaction_count > 0 else 0

        # 平均交易时间 (AnT) 和平均交易长度 (L) 需要先计算总交易长度
        totalTransLength = sum(transaction_lengths)
        AnT = transaction_count / numberOfYears if transaction_count > 0 else 0
        L = totalTransLength / transaction_count if transaction_count > 0 else 0

        # 平均百分比利润 (ApT) 需要总百分比利润，这里简化处理，实际应用中应基于每次交易的盈利计算
        # ApT 目前没有直接计算，需要具体每次交易的利润来累计totalPercentProfit

        # 空闲率 (IdleR)
        IdleR = (merged_df.shape[0] - idle_days - totalTransLength) / merged_df.shape[0] * 100

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
    line_width = 0.7 # 你可以根据需要调整这个值

    plt.figure(figsize=(14, 7))

    df[f"MLP-On-{config['ts_code']}-CAPTIAL"].plot(label=f"MLP-On-{config['ts_code']}-CAPTIAL", marker='o', linewidth=line_width)
    df[f"CNN-On-{config['ts_code']}-CAPTIAL"].plot(label=f"CNN-On-{config['ts_code']}-CAPTIAL", marker='s', linewidth=line_width)
    df[f"TECEC_1-On-{config['ts_code']}-CAPTIAL"].plot(label=f"TECEC_1-On-{config['ts_code']}-CAPTIAL", marker='^', linewidth=line_width)
    df[f"TECEC_2-On-{config['ts_code']}-CAPTIAL"].plot(label=f"TECEC_2-On-{config['ts_code']}-CAPTIAL", marker='x', linewidth=line_width)

    plt.title('Capital Evaluation On SH50')
    plt.xlabel('Date')
    plt.ylabel('Capital Value')
    plt.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(result_dir/f"Capital_Evaluation_on_{config['ts_code']}.png")


    # plot the price of stock 
    plt.figure(figsize=(14, 7))  # 设置图形的尺寸

    # 绘制close价格随日期变化的折线图
    plt.plot(merged_df.index, merged_df['close'], label='Close Price', color='blue')
    # 设置图形标题和坐标轴标签
    plt.title(' SH50')
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


def print_confusion_matrix(config):
    result_dir = Path(
        f"./{config['result_folder']}/{config['ts_code']}/{config['model_name']}")
    epoch_dir = result_dir/'epoch'
    # define the file name list
    file_names = [f"epoch-{i}.csv" for i in range(100, 200)]

    # 初始化一个空的DataFrame用于存储所有数据
    all_data = pd.DataFrame()

    # 循环读取每个CSV文件并合并
    for file_name in file_names:
        data = pd.read_csv(epoch_dir/file_name)
        # 忽略'Date'列
        data = data.drop(columns=['Date'])
        all_data = pd.concat([all_data, data], ignore_index=True)

    # 计算混淆矩阵
    targets = all_data['TARGET']
    predicted = all_data['PREDICTED']

    # 确保TARGET和PREDICTED列只包含0, 1, 2这三个值
    targets = targets.astype(int)
    predicted = predicted.astype(int)

    cm = confusion_matrix(targets, predicted, labels=[0, 1, 2])

    print("混淆矩阵:")
    print(cm)
