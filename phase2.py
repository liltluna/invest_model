from models.dataset import *
from models.models import *
from models.config import *

# train and plot the result of CNN 
model = GraphCNN(input_h=CONFIG['input_h'], input_w=CONFIG['input_w'], num_classes=CONFIG['num_classes']).to(DEVICE)
train(model=model, config=CONFIG, device=DEVICE)
plot_loss_result(config=CONFIG)