import os, model, utils

import torch
import torch.nn as nn
import torch.optim as optim


class AlexNet():
    def __init__(self, num_classes=1000, gpu_id=0, print_freq=10, epoch_print=10):
        self.num_classes = num_classes
        self.gpu = gpu_id
        self.print_freq = print_freq
        self.epoch_print = epoch_print

        torch.cuda.set_device(self.gpu)

        self.loss_function = nn.CrossEntropyLoss().cuda(self.gpu)
        
        self.model = model.AlexNet(self.num_classes).cuda(self.gpu)

        self.train_losses = []
        self.train_acc = []
        self.test_losses = []
        self.test_acc = []
        self.lrs = []
        self.best_acc = 0

    def train(self, train_data, test_data, save, epochs, lr, momentum, weight_decay):
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        
        for epoch in range(epochs):
            if epoch % self.epoch_print == 0: print('Epoch {} Started...'.format(epoch+1))
            for i, (X, y) in enumerate(train_data):
                X, y = X.cuda(self.gpu, non_blocking=True), y.cuda(self.gpu, non_blocking=True)
                output = self.model(X)
                loss = self.loss_function(output, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % self.print_freq == 0:
                    train_acc = 100 * utils.count(output, y) / y.size(0)
                    test_acc, test_loss = self.test(test_data)

                    self.train_losses.append(loss.item())
                    self.train_acc.append(train_acc)
                    self.test_losses.append(test_loss)
                    self.test_acc.append(test_acc)
                    for param_group in optimizer.param_groups:
                        self.lrs.append(param_group['lr'])

                    if epoch % self.epoch_print == 0:
                        state = ('Iteration : {} - Train Loss : {:.4f}, Test Loss : {:.4f}, '
                                 'Train Acc : {:.4f}, Test Acc : {:.4f}').format(i+1, loss.item(), test_loss, train_acc, test_acc)
                        if test_acc > self.best_acc:
                            print()
                            print('*' * 35, 'Best Acc Updated', '*' * 35)
                            print(state)
                            self.best_acc = test_acc
                            if save:
                                torch.save(self.model.state_dict(), './best.pt')
                                print('Saved Best Model')
                        else: print(state)
            scheduler.step(test_loss)

    def test(self, test_data):
        correct, total = 0, 0
        losses = list()

        self.model.eval()
        with torch.no_grad():
            for i, (X, y) in enumerate(test_data):
                X, y = X.cuda(self.gpu, non_blocking=True), y.cuda(self.gpu, non_blocking=True)
                n, crop, _, _, _ = X.shape
                outputs = []
                for j in range(crop):
                    outputs.append(self.model(X[:,j,:,:,:]))
                outputs = torch.stack(outputs)
                output = torch.mean(outputs, dim=0)

                loss = self.loss_function(output, y)
                losses.append(loss.item())

                correct += utils.count(output, y)
                total += y.size(0)
                
        self.model.train()
        return (100*correct/total, sum(losses)/len(losses))