from models.dataset import *
from models.models import *
from models.config import *

# config
CONFIG = {
    "step_len": 7,
    "batch_size": 512,
    "num_epochs": 200,
    "seq_len": 225,
    "lr": 10**-3,
    "d_model": 6,
    "h_num": 2,
    "num_classes": 3,
    "data_folder": 'dataset',
    "result_folder": 'result',
    "experiment_name": "runs/tmodel",
    "input_w": 15,
    "input_h": 15,
    "label_method": "graph_label",
    "ts_code": "000700.SZ",
    "model_name": "CNN",
}

# Define the device
device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
# device = "cpu"
print("Using device:", device)
if (device == 'cuda:0'):
    print(f"Device name: {torch.cuda.get_device_name(device.index)}")
    print(
        f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
else:
    print("NOTE: If you have a GPU, consider using it for training.")
device = torch.device(device)


# train 
model = GraphCNN(input_h=CONFIG['input_h'], input_w=CONFIG['input_w'], num_classes=CONFIG['num_classes']).to(device)

train(model=model, config=CONFIG, device=device)
plot_result(config=CONFIG)