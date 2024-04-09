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
from torch.nn.functional import one_hot


class PolyLoss(torch.nn.Module):
    """
    Implementation of poly loss.
    Refers to `PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions (ICLR 2022)
    <https://arxiv.org/abs/2204.12511>
    """

    def __init__(self, num_classes=2, epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        self.num_classes = num_classes

    def forward(self, output, target):
        ce = self.criterion(output, target)
        pt = one_hot(target, num_classes=self.num_classes) * self.softmax(output)

        return (ce + self.epsilon * (1.0 - pt.sum(dim=-1))).mean()

class CompositeLoss(nn.Module):
    def __init__(self, num_classes, weight_criterion1=1.0, weight_criterion2=0.5):
        super(CompositeLoss, self).__init__()
        self.criterion1 = nn.CrossEntropyLoss(reduction='mean')
        self.criterion2 = PolyLoss(num_classes=num_classes)
        self.weight_criterion1 = weight_criterion1
        self.weight_criterion2 = weight_criterion2

    def forward(self, input, target):
        loss1 = self.criterion1(input, target)
        loss2 = self.criterion2(input, target)
        composite_loss = self.weight_criterion1 * loss1 + self.weight_criterion2 * loss2
        return composite_loss

def initialize(args):
    model = CovidNet(args.bnd, args.bna, args.classes)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()

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

    return model, optimizer, training_generator, test_generator


def train(args, model, trainloader, optimizer, epoch):
    model.train()
    # criterion1 = nn.CrossEntropyLoss(reduction='elementwise_mean')
    # criterion2 = PolyLoss(num_classes=args.classes)
    criterion = CompositeLoss(num_classes=args.classes, weight_criterion1=1.0, weight_criterion2=0.5)


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

        if len(features) == 32 and args.cont:
            temperature = 0.05
            cont_loss_func = losses.NTXentLoss(temperature)
            cont_loss = cont_loss_func(features, labels)
            loss = criterion(output, labels) + cont_loss
        else:
            loss = criterion(output, labels)

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


def validation(args, model, testloader, epoch, mode='test'):
    conf_matrix = torch.zeros(args.classes, args.classes).cuda()
    model.eval()
    # criterion1 = nn.CrossEntropyLoss(reduction='elementwise_mean')
    # criterion2 = PolyLoss(num_classes=args.classes)
    # criterion = criterion1 + 0.5*criterion2
    criterion = CompositeLoss(num_classes=args.classes, weight_criterion1=1.0, weight_criterion2=0.5)
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
