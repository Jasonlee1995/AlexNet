import torch

torch.manual_seed(0)


def count(output, target):
    with torch.no_grad():
        predict = torch.argmax(output, 1)
        correct = (predict == target).sum().item()
        return correct


def save_checkpoint(num_classes, epoch, state):
    filename = './checkpoints/checkpoint_' + '0'*(5-len(str(num_classes))) + str(num_classes)
    filename += '_' + '0'*(3-len(str(epoch))) + str(epoch)
    filename += '.pth.tar'
    torch.save(state, filename)
