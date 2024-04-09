# coding=UTF-8
import torch
import os
import torch.nn as nn
from util import Metrics, print_summary
from metric import accuracy, top_k_acc
from dataset import COVID_CT_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from skimage.segmentation import slic, mark_boundaries
from model3 import *
import torch.optim as optim
from pytorch_metric_learning import losses
from ISDA import ISDALoss
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class Full_layer(torch.nn.Module):
    '''explicitly define the full connected layer'''

    def __init__(self, feature_num, class_num):
        super(Full_layer, self).__init__()
        self.class_num = class_num
        self.fc = nn.Linear(feature_num, class_num)

    def forward(self, x):
        x = self.fc(x)
        return x

def initialize(args):

    model = CovidNet(args.bnd, args.bna, args.classes)

    fc = Full_layer(model.feature_num, args.classes)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    if args.cuda:
        model.cuda()

        fc.cuda()

    train_params = {'batch_size': args.batch_size,
                    'shuffle': True,
                    'num_workers': 0}
    test_params = {'batch_size': args.batch_size,
                   'shuffle': False,
                   'num_workers': 0}

    train_loader = COVID_CT_Dataset(args, mode='train', n_classes=args.classes, dataset_path=args.dataset,
                                    dim=(224, 224))
    test_loader = COVID_CT_Dataset(args, mode='test', n_classes=args.classes, dataset_path=args.dataset,
                                   dim=(224, 224))

    training_generator = DataLoader(train_loader, **train_params)
    test_generator = DataLoader(test_loader, **test_params)



    return model, fc, optimizer, training_generator, test_generator


def train(args, model, fc, trainloader, optimizer, epoch, isda_criterion):
    model.train()
    fc.train()

    criterion = nn.CrossEntropyLoss(reduction='elementwise_mean')

    metrics = Metrics('')
    metrics.reset()
    batch_idx = 0

    for input_tensors in tqdm(trainloader):
        batch_idx = batch_idx + 1
        optimizer.zero_grad()
        input_data, target, site = input_tensors
        if args.cuda:
            input_data = input_data.cuda()
            target = target.cuda()

        ucsd_input = torch.from_numpy(np.array([]))
        new_input = torch.from_numpy(np.array([]))
        ucsd_label = torch.from_numpy(np.array([]))
        new_label = torch.from_numpy(np.array([]))

        # for i in range(len(input_data)):
        #     if site[i] == 'ucsd':
        #         if len(ucsd_input) == 0:
        #             ucsd_input = input_data[i].unsqueeze(0)
        #             ucsd_label = torch.from_numpy(np.array([target[i]]))
        #         else:
        #             ucsd_input = torch.cat((ucsd_input, input_data[i].unsqueeze(0)))
        #             ucsd_label = torch.cat((ucsd_label, torch.from_numpy(np.array([target[i]]))))
        #     else:
        #         if len(new_input) == 0:
        #             new_input = input_data[i].unsqueeze(0)
        #             new_label = torch.from_numpy(np.array([target[i]]))
        #         else:
        #             new_input = torch.cat((new_input, input_data[i].unsqueeze(0)))
        #             new_label = torch.cat((new_label, torch.from_numpy(np.array([target[i]]))))
        for i in range(len(input_data)):
            if site[i] == 'ucsd':
                if ucsd_input.nelement() == 0:
                    ucsd_input = input_data[i].unsqueeze(0)
                    ucsd_label = target[i].unsqueeze(0)
                else:
                    ucsd_input = torch.cat((ucsd_input, input_data[i].unsqueeze(0)), 0)
                    ucsd_label = torch.cat((ucsd_label, target[i].unsqueeze(0)), 0)
            else:
                if new_input.nelement() == 0:
                    new_input = input_data[i].unsqueeze(0)
                    new_label = target[i].unsqueeze(0)
                else:
                    new_input = torch.cat((new_input, input_data[i].unsqueeze(0)), 0)
                    new_label = torch.cat((new_label, target[i].unsqueeze(0)), 0)

        if len(ucsd_input) > 1:
            ucsd_output, ucsd_features = model(ucsd_input, 'ucsd')
        if len(new_input) > 1:
            new_output, new_features = model(new_input, 'new')

        if len(ucsd_input) > 1 and len(new_input) > 1:
            output = torch.cat((ucsd_output, new_output))
            labels = torch.cat((ucsd_label, new_label)).cuda()
            features = torch.cat((ucsd_features, new_features))
        elif len(ucsd_input) > 1 and len(new_input) < 2:
            output = ucsd_output
            labels = ucsd_label.cuda()
            features = ucsd_features
        else:
            output = new_output
            labels = new_label.cuda()
            features = new_features

        if len(output) != len(labels):
            continue

        # ratio = args.lambda_0*(epoch/args.nEpochs)
        ratio = args.lambda_0
        isda_loss,_ = isda_criterion(model, fc, input_data, labels, ratio, site) # Compute ISDA loss


        if len(features) == 32 and args.cont:
            temperature = 0.05
            cont_loss_func = losses.NTXentLoss(temperature)
            cont_loss = cont_loss_func(features, labels)


            loss = criterion(output, labels) + cont_loss + isda_loss
            # loss = criterion(output, labels)
        else:

            loss = criterion(output, labels) + isda_loss
            # loss = criterion(output, labels)

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        correct, total, acc = accuracy(output, labels)
        top1_correct = top_k_acc(output, labels, k=1)
        top3_correct = top_k_acc(output, labels, k=2)

        metrics.update({'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc,
                        'top1_correct': top1_correct, 'top3_correct': top3_correct})

    print_summary(args, epoch, metrics, mode="train")
    return metrics


def validation(args, model, fc, testloader, epoch, mode='test'):
    conf_matrix = torch.zeros(args.classes, args.classes).cuda()
    model.eval()
    fc.eval()
    criterion = nn.CrossEntropyLoss()

    metrics = Metrics('')
    metrics.reset()
    batch_idx = 0
    ucsd_correct_total = 0
    sars_correct_total = 0
    ucsd_test_total = 0
    sars_test_total = 0
    with torch.no_grad():
        for input_tensors in tqdm(testloader):
            batch_idx = batch_idx + 1
            input_data, target, site = input_tensors
            if args.cuda:
                input_data = input_data.cuda()
                target = target.cuda()

            ucsd_input = torch.from_numpy(np.array([]))
            new_input = torch.from_numpy(np.array([]))
            ucsd_label = torch.from_numpy(np.array([]))
            new_label = torch.from_numpy(np.array([]))

            # for i in range(len(input_data)):
            #     if site[i] == 'ucsd':
            #         if len(ucsd_input) == 0:
            #             ucsd_input = input_data[i].unsqueeze(0)
            #             ucsd_label = torch.from_numpy(np.array([target[i]]))
            #         else:
            #             ucsd_input = torch.cat((ucsd_input, input_data[i].unsqueeze(0)))
            #             ucsd_label = torch.cat((ucsd_label, torch.from_numpy(np.array([target[i]]))))
            #     else:
            #         if len(new_input) == 0:
            #             new_input = input_data[i].unsqueeze(0)
            #             new_label = torch.from_numpy(np.array([target[i]]))
            #         else:
            #             new_input = torch.cat((new_input, input_data[i].unsqueeze(0)))
            #             new_label = torch.cat((new_label, torch.from_numpy(np.array([target[i]]))))
            for i in range(len(input_data)):
                if site[i] == 'ucsd':
                    if ucsd_input.nelement() == 0:
                        ucsd_input = input_data[i].unsqueeze(0)
                        ucsd_label = target[i].unsqueeze(0)
                    else:
                        ucsd_input = torch.cat((ucsd_input, input_data[i].unsqueeze(0)), dim=0)
                        ucsd_label = torch.cat((ucsd_label, target[i].unsqueeze(0)), dim=0)
                else:
                    if new_input.nelement() == 0:
                        new_input = input_data[i].unsqueeze(0)
                        new_label = target[i].unsqueeze(0)
                    else:
                        new_input = torch.cat((new_input, input_data[i].unsqueeze(0)), dim=0)
                        new_label = torch.cat((new_label, target[i].unsqueeze(0)), dim=0)

            if len(ucsd_input) > 1:
                ucsd_output, ucsd_features = model(ucsd_input, 'ucsd')
                ucsd_correct, ucsd_total, ucsd_acc = accuracy(ucsd_output, ucsd_label.cuda())
                ucsd_correct_total += ucsd_correct
                ucsd_test_total += ucsd_total

            if len(new_input) > 1:
                new_output, new_features = model(new_input, 'new')
                sars_correct, sars_total, sars_acc = accuracy(new_output, new_label.cuda())
                sars_correct_total += sars_correct
                sars_test_total += sars_total

            if len(ucsd_input) > 1 and len(new_input) > 1:
                output = torch.cat((ucsd_output, new_output))
                labels = torch.cat((ucsd_label, new_label)).cuda()
                features = torch.cat((ucsd_features, new_features))
            elif len(ucsd_input) > 1 and len(new_input) < 2:
                output = ucsd_output
                labels = ucsd_label.cuda()
            else:
                output = new_output
                labels = new_label.cuda()

            loss = criterion(output, labels)
            preds = torch.argmax(output, dim=1)
            for t, p in zip(target.view(-1), preds.view(-1)):
                conf_matrix[t.long(), p.long()] += 1

            correct, total, acc = accuracy(output, labels)

            # top k acc
            top1_correct = top_k_acc(output, labels, k=1)
            top3_correct = top_k_acc(output, labels, k=2)

            metrics.update({'correct': correct, 'total': total, 'loss': loss.item(), 'accuracy': acc,
                            'top1_correct': top1_correct, 'top3_correct': top3_correct})

    print_summary(args, epoch, metrics, mode="test")

    return metrics, conf_matrix, ucsd_correct_total, sars_correct_total, ucsd_test_total, sars_test_total
