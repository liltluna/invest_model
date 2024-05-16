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
