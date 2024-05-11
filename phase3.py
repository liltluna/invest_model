
from models.dataset import *
from models.models import *
from models.process import *
from pathlib import Path
from tqdm import tqdm


def get_model(config):
    model = GraphCNN(input_h=config['input_h'], input_w=config['input_w'], num_classes=config['num_classes'])
    return model


config = get_universial_config()

# Define the device
device = "cuda:0" if torch.cuda.is_available(
) else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
# device = "cpu"
print("Using device:", device)
if (device == 'cuda:0'):
    print(f"Device name: {torch.cuda.get_device_name(device.index)}")
    print(
        f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
elif (device == 'mps'):
    print(f"Device name: <mps>")
else:
    print("NOTE: If you have a GPU, consider using it for training.")
device = torch.device(device)

# Make sure the weights folder exists
Path(f"./{config['model_folder']}").mkdir(parents=True, exist_ok=True)

train_dataloader, val_dataloader = get_ds(config)
model = get_model(config).to(device)
# Tensorboard
# writer = SummaryWriter(config['experiment_name'])
# writer = None
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

# If the user specified a model to preload before training, load it
initial_epoch = 0
global_step = 0
preload = config['preload']
model_filename = None
if model_filename:
    print(f'Preloading model {model_filename}')
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])
    initial_epoch = state['epoch'] + 1
    optimizer.load_state_dict(state['optimizer_state_dict'])
    global_step = state['global_step']
else:
    print('No model to preload, starting from scratch')

loss_fn = nn.CrossEntropyLoss().to(device)

if not os.path.exists('./result'):
    os.mkdir('./result')

fieldnames = ['epoch', 'loss', 'acc']
with open(f'./result/statistic.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

for epoch in range(initial_epoch, config['num_epochs']):
    torch.cuda.empty_cache()
    model.train()
    batch_iterator = tqdm(
        train_dataloader, desc=f"Processing Epoch {epoch:02d}")
    for batch in batch_iterator:

        nn_input = batch['nn_input'].to(device)  # (b, seq_len)
        encoder_mask = None
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
    accuracy = run_validation(model, val_dataloader, config['step_len'], device, lambda msg: batch_iterator.write(
        msg), global_step, writer=None, epoch=epoch)
    with open(f'./result/statistic.csv', 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # 将数据写入
        writer.writerow({'epoch': epoch, 'loss': loss.item(), 'acc': accuracy})