import torch
import torch.nn as nn
from torch.nn import Parameter
import torchvision
import torch.optim as optim
import pickle
import numpy as np
from cigcn import CI_GCN
from src.models.tresnet.tresnet import TResnetM, TResnetL, TResnetXL

model_dict = {'ADD_GCN': CI_GCN}

def get_model_tresnet(start,end,device):
        """
        Create model tresnet
        """
        pretrained_path = "./src/pretrained_tresnet/tresnet_m_224_21k.pth"
        num_classes = end - start
        model_name = 'tresnet_m'
        model = create_model(model_name, num_classes)
        state = torch.load(pretrained_path, map_location='cpu')
        state = {(k if 'body.' not in k else k[5:]): v for k, v in state['model'].items()}
        filtered_dict = {k: v for k, v in state.items() if
                        (k in model.state_dict() and 'head.fc' not in k)}

        model.load_state_dict(filtered_dict, strict=False)
        model = model_dict['ADD_GCN'](model, start, end)
        return model

def create_model(model_name, num_classes):
    """Create a model, with model_name and num_classes
    """
    model_params = {'num_classes': num_classes}

    if model_name=='tresnet_m':
        model = TResnetM(model_params)
    elif model_name=='tresnet_l':
        model = TResnetL(model_params)
    elif model_name=='tresnet_xl':
        model = TResnetXL(model_params)
    
    else:
        print("model: {} not found !!".format(model_name))
        exit(-1)

    return model

