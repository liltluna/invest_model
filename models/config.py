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


def run_validation(model, validation_ds, config, device, print_msg, global_step, writer, epoch, num_examples=1):
    fieldnames = ['Date', 'TARGET', 'PREDICTED']

    result_dir = Path(f"./{config['result_folder']}/{config['ts_code']}/{config['model_name']}")
    result_dir.mkdir(parents=True, exist_ok=True)
    epoch_path = result_dir/f'epoch/epoch-{epoch}.csv'
    with open(epoch_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    model.eval()
    output_count = 0
    match_count = 0

    with torch.no_grad():
        for batch in validation_ds:
            output_count += 1
            nn_input = batch["nn_input"].to(device)  # (b, seq_len)
            # check that the batch size is 1
            assert nn_input.size(0) == 1, "Batch size must be 1 for validation"
            # Run the tensors through the encoder, decoder and the projection layer
            nn_output = model.forward(nn_input)
            _, predicted_labels = torch.max(nn_output, dim=1)
            target = batch['label'][0]
            date = batch['date'][0]
            if target == predicted_labels:
                match_count += 1
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

            # accuracy_metric.update(preds=proj_output, target=target_output)
            # if count == num_examples:
            #     print_msg('-'*console_width)
            #     break
        # acc = accuracy_metric.compute().item()
        accuracy = match_count / len(validation_ds)
        print(f'accuracy: {accuracy}')

    return accuracy


def train(model, device, config):
    result_dir = Path(f"./{config['result_folder']}/{config['ts_code']}/{config['model_name']}")
    epoch_dir = result_dir/'epoch'
    epoch_dir.mkdir(parents=True, exist_ok=True)

    # Make sure the weights folder exists
    # Path(f"./{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader = get_ds(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0

    # if model_filename:
    #     print(f'Preloading model {model_filename}')
    #     state = torch.load(model_filename)
    #     model.load_state_dict(state['model_state_dict'])
    #     initial_epoch = state['epoch'] + 1
    #     optimizer.load_state_dict(state['optimizer_state_dict'])
    #     global_step = state['global_step']
    # else:

    print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss().to(device)

    fieldnames = ['epoch', 'loss', 'acc']
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

            # Log the loss
            # writer.add_scalar('train loss', loss.item(), global_step)
            # writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        accuracy = run_validation(model=model, validation_ds=val_dataloader, global_step=global_step, device=device,print_msg= lambda msg: batch_iterator.write(
            msg), writer=None, epoch=epoch, config=config)
        
        with open(result_dir/"statistic.csv", 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # 将数据写入
            writer.writerow(
                {'epoch': epoch, 'loss': loss.item(), 'acc': accuracy})

def plot_result(config):

    window = 10
    statistic_dir = Path(f"./{config['result_folder']}/{config['ts_code']}/{config['model_name']}")
    statistic_path = statistic_dir/"statistic.csv"
    # window = 10
    # 步骤1: 加载CSV文件中的数据
    data = pd.read_csv(statistic_path)

    # 计算损失和准确率的移动平均值
    data['Smoothed Loss'] = data['loss'].rolling(window=window).mean()
    data['Smoothed Accuracy'] = data['acc'].rolling(window=window).mean()  # 假设'accuracy'是准确率的列名

    # 提取epoch、平滑后的loss和accuracy数据
    epochs = data['epoch']
    losses = data['Smoothed Loss']
    accuracies = data['Smoothed Accuracy']

    # 步骤3: 绘制loss和accuracy曲线
    plt.figure(figsize=(10, 5))  # 设置图形大小

    # 绘制loss曲线
    plt.plot(epochs, losses, label='Loss', color='blue')  # 绘制平滑后的loss曲线，颜色设为蓝色

    # 绘制accuracy曲线
    plt.plot(epochs, accuracies, label='Accuracy', color='green')  # 注意原注释中的颜色被修正为绿色

    # 添加标题和坐标轴标签
    plt.title(f"{config['model_name']} On {config['ts_code']}")
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(False)  # 原本为False，这里假设你想显示网格，根据需要可以调整

    # 直接保存图像，而不是显示
    plt.savefig(statistic_dir/f"{config['model_name']}_On_{config['ts_code']}.png")
    