from model import *
from utils import *
import os

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)


class AlexNet():
    def __init__(self, num_classes=1000, gpu_id=0, print_freq=10, epoch_print=10, epoch_save=50):

        self.num_classes = num_classes
        self.gpu = gpu_id
        self.print_freq = print_freq
        self.epoch_print = epoch_print
        self.epoch_save = epoch_save

        torch.cuda.set_device(self.gpu)

        self.loss_function = nn.CrossEntropyLoss().cuda(self.gpu)

        print('=> Create AlexNet')

        model = alexnet(self.num_classes)
        self.model = model.cuda(self.gpu)

        self.train_losses = list()
        self.train_acc = list()
        self.test_losses = list()
        self.test_acc = list()


    def train(self, train_data, test_data, resume=False, save=False, start_epoch=0, epochs=90,
              lr=0.01, momentum=0.9, weight_decay=0.0005, milestones=False):
        # Model to Train Mode
        self.model.train()

        # Set Optimizer and Scheduler
        optimizer = optim.SGD(self.model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
        if milestones:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [epochs//2, epochs*3//4], gamma=0.1)

        # Optionally Resume from Checkpoint
        if resume:
            if os.path.isfile(resume):
                print('=> Load checkpoint from {}'.format(resume))
                loc = 'cuda:{}'.format(self.gpu)
                checkpoint = torch.load(resume, map_location=loc)

                self.model.load_state_dict(checkpoint['state_dict'])

                start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                print('=> Loaded checkpoint from {} with epoch {}'.format(resume, checkpoint['epoch']))
            else:
                print('=> No checkpoint found at {}'.format(resume))

        # Train
        for epoch in range(start_epoch, epochs):
            if epoch % self.epoch_print == 0:
                print('Epoch {} Started...'.format(epoch+1))
            for i, (X, y) in enumerate(train_data):
                X, y = X.cuda(self.gpu, non_blocking=True), y.cuda(self.gpu, non_blocking=True)
                output = self.model(X)
                loss = self.loss_function(output, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % self.print_freq == 0:
                    train_acc = 100 * count(output, y) / y.size(0)
                    test_acc, test_loss = self.test(test_data)

                    self.train_losses.append(loss.item())
                    self.train_acc.append(train_acc)
                    self.test_losses.append(test_loss)
                    self.test_acc.append(test_acc)

                    self.model.train()

                    if epoch % self.epoch_print == 0:
                        print('Iteration : {} - Train Loss : {:.2f}, Test Loss : {:.2f}, '
                              'Train Acc : {:.2f}, Test Acc : {:.2f}'.format(i+1, loss.item(), test_loss,
                                                                             train_acc, test_acc))

            scheduler.step()
            if save and (epoch % self.epoch_save == 0):
                save_checkpoint(self.num_classes, epoch, state={'epoch': epoch+1,
                                                                'state_dict':self.model.state_dict(),
                                                                'optimizer':optimizer.state_dict(), 'scheduler':scheduler})


    def test(self, test_data):
        correct, total = 0, 0
        losses = list()

        # Model to Eval Mode
        self.model.eval()

        # Test
        with torch.no_grad():
            for i, (X, y) in enumerate(test_data):
                X, y = X.cuda(self.gpu, non_blocking=True), y.cuda(self.gpu, non_blocking=True)
                output = self.model(X)

                loss = self.loss_function(output, y)
                losses.append(loss.item())

                correct += count(output, y)
                total += y.size(0)

        return (100*correct/total, sum(losses)/len(losses))
