import torch
import csv
import torch
from models.dataset import *
from torch.utils.data import DataLoader

# config 
def get_universial_config():
    return {
        "step_len" : 7, 
        "batch_size": 512,
        "num_epochs": 200,
        "seq_len": 225, 
        "lr": 10**-3,
        "d_model": 6,
        "h_num": 2,
        "num_classes": 3,
        "data_folder": 'dataset',
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "experiment_name": "runs/tmodel",
        "input_w": 15, 
        "input_h":15,
    }


# get dataloader
def get_ds(config):
    ds_raw = BasicGraphDataset('./dataset/output_phase2_graph_label_1_000016.SH_train.csv', 'train')
    ds_raw_test = BasicGraphDataset('./dataset/output_phase2_graph_label_1_000016.SH_test.csv', 'test')

    train_ds = GraphDataset(ds=ds_raw, seq_len=config['seq_len'], num_classes=config["num_classes"])
    val_ds = GraphDataset(ds=ds_raw_test, seq_len=config['seq_len'], num_classes=config["num_classes"])

    train_dataloader = DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader

def run_validation(model, validation_ds, max_len, device, print_msg, global_step, writer, epoch, num_examples=1):
    fieldnames = ['Date', 'TARGET', 'PREDICTED']
    with open(f'./result/cnn_with_1_encoder-epoch{epoch}.csv', 'w', newline='', encoding='utf-8') as csvfile:
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
            with open(f'./result/cnn_with_1_encoder-epoch{epoch}.csv', 'a', newline='', encoding='utf-8') as csvfile:
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
