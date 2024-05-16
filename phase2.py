from models.models import *
from models.process import *
from models.config import *
from models.vision import *

# CNN
model = GraphCNN(input_h=CONFIG['input_h'], input_w=CONFIG['input_w'],
                 num_classes=CONFIG['num_classes']).to(DEVICE)
train(model=model, config=CONFIG, device=DEVICE)
plot_loss_result(config=CONFIG)

# MLP
CONFIG['model_name'] = 'MLP'
model = MLP(input_h=CONFIG['input_h'], input_w=CONFIG['input_w'],
            num_classes=CONFIG['num_classes']).to(DEVICE)
train(model=model, config=CONFIG, device=DEVICE)
plot_loss_result(config=CONFIG)

# TECEC_1
CONFIG['model_name'] = 'TECEC_1'
model = TECEC_1(input_h=CONFIG['input_h'], input_w=CONFIG['input_w'],
                num_classes=CONFIG['num_classes']).to(DEVICE)
train(model=model, config=CONFIG, device=DEVICE)
plot_loss_result(config=CONFIG)

# TECEC_2
CONFIG['model_name'] = 'TECEC_2'
model = TECEC_2(input_h=CONFIG['input_h'], input_w=CONFIG['input_w'],
                num_classes=CONFIG['num_classes']).to(DEVICE)
train(model=model, config=CONFIG, device=DEVICE)
plot_loss_result(config=CONFIG)

# RNN
CONFIG['model_name'] = 'RNN'
model = RNN().to(DEVICE)
train(model=model, config=CONFIG, device=DEVICE)
plot_loss_result(config=CONFIG)

# LSTM
CONFIG['model_name'] = 'LSTM'
model = LSTM().to(DEVICE)
train(model=model, config=CONFIG, device=DEVICE)
plot_loss_result(config=CONFIG)

# implement the financial evaluation
plot_finicial_evalutaion_comparation(config=CONFIG)
