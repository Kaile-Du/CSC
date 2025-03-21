from src.helper_functions.IncrementalDataset import build_dataset, build_loader
import torchvision.transforms as transforms
from src.helper_functions.helper_functions import CutoutPIL, add_weight_decay
from randaugment import RandAugment
import copy
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from evaluation_metrics import prf_cal, mAP
from utils import get_model_tresnet
from buffer import Buffer
from sample_proto import icarl_sample_protos, random_sample_protos
from PIL import Image
import json
from collections import Counter
import pandas as pd
torch.cuda.set_device(5)

class CSC_MLCIL:
    def __init__(self, args):
        self.task_id = 0
        self.total_map = 0
        self.lr=args.lr

        "VOC"
        self.base_classes = args.base_classes
        self.task_size = args.task_size
        self.total_classes = args.total_classes
        self.dataset_name = args.dataset_name
        self.root_dir = args.root_dir
        "VOC"

        self.image_size = args.input_size
        self.start = 0
        self.end = self.base_classes
        self.num_epochs = args.epochs
        self.device = torch.device('cuda:5' if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MultiLabelSoftMarginLoss()
        self.alpha = args.alpha
        self.beta = args.beta

        self.model = get_model_tresnet(self.start,self.end,self.device)
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        self.train_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            CutoutPIL(cutout_factor=0.5),
            RandAugment(),
            transforms.ToTensor(),
        ])
        self.val_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.pre_model = None

        

        # parameters of replay
        self.replay = args.replay
        self.buffer_size = args.buffer_size
        self.replay_alpha = args.replay_alpha
        self.buffer = Buffer(self.buffer_size, self.device)
        self.num_protos = args.num_protos
        self.sample_method = args.sample_method
        self.old_dataset = []
        self.image_id_set = set()
        self.herding_index = 0


    def network_expansion(self, low_range, high_range):
        in_features = self.model.fc.weight.size(0)
        out_features = self.model.fc.weight.size(1)
        weight = self.model.fc.weight.data
        self.model.fc = nn.Conv2d(out_features, in_features + self.task_size, (1,1), bias=False).to(self.device)
        self.model.fc.weight.data[:low_range,:] = weight
        for name, layer in self.model.gcn.named_children():
            if name == 'general_adj':
                A_s_weights = layer[0].weight.data
        new_A_s = nn.Conv1d(high_range, high_range, 1, bias=False)
        new_A_s.weight.data[:low_range, :low_range, :] = A_s_weights
        new_A_s = new_A_s.to(self.device)
        for name, layer in self.model.gcn.named_children():
            if name == 'general_adj':
                layer[0] = new_A_s
        for name, layer in self.model.gcn.named_children():
            if name == 'conv_create_co_mat':
                in_features = layer.weight.size(1)
                conv_create_co_mat_weights = layer.weight.data
        new_conv_create_co_mat = nn.Conv1d(in_features, high_range, 1)
        new_conv_create_co_mat.weight.data[:low_range, :, :] = conv_create_co_mat_weights
        new_conv_create_co_mat = new_conv_create_co_mat.to(self.device)
        self.model.gcn.conv_create_co_mat = new_conv_create_co_mat

        in_features = self.model.last_linear.weight.size(0)
        out_features = self.model.last_linear.weight.size(1)
        weight = self.model.last_linear.weight.data
        self.model.last_linear = nn.Conv1d(out_features, in_features + self.task_size, 1).to(self.device)
        self.model.last_linear.weight.data[:low_range,:] = weight

        new_mask_mat = nn.Parameter(torch.eye(high_range).float().to(self.device))
        new_mask_mat.data[:low_range,:low_range] = self.model.mask_mat.data
        self.model.mask_mat = new_mask_mat


    def get_train_dataloader(self, low_range, high_range):
        train_dataset_without_old = build_dataset(self.dataset_name, self.root_dir, low_range, high_range,
                                          phase='train', transform=self.train_transforms)
        train_dataset = train_dataset_without_old
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= self.batch_size, num_workers=self.num_workers, drop_last=True)
        return train_loader, train_dataset
    
    def get_val_seen_dataloader(self, high_range):
        val_dataset_seen = build_dataset(self.dataset_name, self.root_dir, 0, high_range, phase='val',
                                             transform=self.val_transforms)
        val_loader_seen = build_loader(val_dataset_seen, self.batch_size, self.num_workers, phase='val')
        return val_loader_seen

    def train_test(self):
        counter = Counter()
        base_stage = [(0, self.base_classes)] 
        incremental_stages = base_stage + [
            (low, low + self.task_size) for low in range(self.base_classes, self.total_classes, self.task_size)]
        self.model.train()  
        for low_range, high_range in incremental_stages:
            train_loader, train_dataset = self.get_train_dataloader(low_range, high_range)
            val_loader_seen = self.get_val_seen_dataloader(high_range)
            if(int(self.task_id) != 0):
                self.network_expansion(low_range, high_range)
            for epoch in range(self.num_epochs):
                print('epoch: ', epoch)
                for i, (inputs, labels) in enumerate(train_loader):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    inputs = inputs.float()
                    labels = labels.float()
                    labels_tr = labels.clone()
                    self.optimizer.zero_grad()
                    # forward
                    with torch.set_grad_enabled(True):
                        outputs1, outputs2, _= self.model(inputs)
                        outputs = (outputs1 + outputs2) / 2
                        loss = self.criterion(outputs[:, low_range : high_range], labels_tr[:, low_range : high_range]) 
                        if(self.task_id > 0):
                            dist_outputs1, dist_outputs2, _ = self.pre_model(inputs)
                            dist_target = torch.sigmoid((dist_outputs1+dist_outputs2)/2)
                            dist_logits = outputs[:,:low_range]
                            dist_loss = self.criterion(dist_logits, dist_target)
                            loss  = self.alpha * loss
                            loss += (1-self.alpha)*dist_loss

                        '''Max-Entropy'''
                        if(self.task_id > 0):
                            outputs_ent = torch.sigmoid(outputs[:,  : high_range])
                            loss_ent = -self.beta * torch.mean(torch.sum(outputs_ent *
                                                (torch.log(outputs_ent + 1e-5)), 1))
                            loss += loss_ent
                        '''Max-Entropy'''
                        loss.backward()
                        self.optimizer.step()

            print("\nTraining finished.")
                
            self.model.eval()
            for i, (inputs, labels) in enumerate(val_loader_seen):
                if inputs.size(0) > 1 : 
                    labels = labels[:, :high_range] 
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    inputs = inputs.float()
                    labels = labels.float()
                    self.optimizer.zero_grad()

                with torch.set_grad_enabled(False):
                    outputs1, outputs2, _= self.model(inputs)
                    outputs = (outputs1+outputs2)/2
                    if i == 0:
                        outputs_test = outputs
                        labels_test = labels
                    else:
                        outputs_test = torch.cat((outputs_test, outputs), 0)
                        labels_test = torch.cat((labels_test, labels), 0)
            outputs_test = torch.sigmoid(outputs_test)
            mAP_score, _ = mAP(labels_test.to(torch.device("cpu")).numpy(),
                                                       outputs_test.to(torch.device("cpu")).numpy())
                                                       
            print('Task_id: ', self.task_id)
            print('Test:')
            print('mAP',mAP_score)
            self.total_map = self.total_map + mAP_score
            print("mean_map",self.total_map/(self.task_id+1))   
            CP, CR, CF1, OP, OR, OF1 = prf_cal(outputs_test.to(torch.device("cpu")),
                                            labels_test.to(torch.device("cpu")), outputs_test)                       
            print('CP',CP)
            print('CR',CR)
            print('CF1',CF1)
            print('OP',OP)
            print('OR',OR)
            print('OF1',OF1)
            print("\nTest finished.")



            self.pre_model = copy.deepcopy(self.model)
            self.task_id = self.task_id + 1
            self.start = high_range
            self.end = high_range + self.task_size
            # torch.save(self.model.state_dict(),
            #            './checkpoint2_csc/' + "b" + str(self.base_classes) + 'c' + str(self.task_size) + "/"
            #            + "task" + str(self.task_id) + '.pth')
