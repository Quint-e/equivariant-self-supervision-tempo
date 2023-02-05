# -*- coding: utf-8 -*-
"""
@author: Elio Quinton
python3.8
"""
from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 10e-8
# EPS = 0

def general_crossentropy(outputs, targets, reduction='mean'):
    '''Outputs and targets should have the same dimension: [batch_size,*]'''
    losses = torch.sum(targets*torch.log(outputs + EPS),dim=1) # Loss foreach element in the batch
    if reduction=='mean':
        loss = -torch.mean(losses)
    elif reduction=='sum':
        loss = -torch.sum(losses)
    elif reduction=='none':
        loss = -losses
    else:
        raise ValueError("reduction must be one of {'mean', 'sum', 'none'}")
    return loss

class XentBoeck(torch.nn.Module):
    '''Boeck cross-entropy loss'''
    def __init__(self, boeck_window = torch.Tensor([[[0.1, 0.2, 0.4, 0.2, 0.1]]]), reduction='mean', device='cpu'):
        super(XentBoeck, self).__init__()
        self.device = device
        self.xent = nn.CrossEntropyLoss(reduction=reduction)
        # self.xent = nn.NLLLoss(reduction=reduction)
        self.boeck_window = boeck_window
        assert torch.sum(self.boeck_window).item() == 1.0, 'boeck window should sum to one, but got {}'.format(self.boeck_window)
        assert self.boeck_window.shape[2]%2 != 0, 'Boeck window should be of odd length, but got {}'.format(len(self.boeck_window))

    def generate_boeck_target(self, preds: torch.Tensor,  labels_cls_idx: torch.Tensor) -> torch.Tensor:
        '''
        Generate Boeck target by convolving one-hot vectors with the boeck window
        labels_cls_idx of shape (batch_size)'''
        conv = nn.Conv1d(1, 1, self.boeck_window.shape[2], bias=False, padding='same')
        conv.weight = torch.nn.Parameter(self.boeck_window)
        one_hot = torch.zeros(labels_cls_idx.shape[-1], 1, preds.shape[-1]) #include channel dimension for conv
        for i, idx in enumerate(labels_cls_idx):
            one_hot[i, 0, int(idx.item())] = 1.0
        return torch.squeeze(conv(one_hot)) #remove channel dimension to output shape (batch_size, n_classes)

    def forward(self, preds, labels_cls_idx):
        boeck_target = self.generate_boeck_target(preds, labels_cls_idx).to(self.device)
        loss = self.xent(preds, boeck_target)
        return loss




if __name__ == "__main__":
    # Test

    def test_onehot():
        print('TEST one-hot...')
        batch_size = 16
        output_dim = 10

        outputs = torch.randn(batch_size, output_dim)
        # outputs = torch.clamp(outputs, min=0.0)
        print('outputs', outputs.shape, outputs)
        # outputs = F.softmax(outputs, dim=1)
        s = torch.sum(outputs, 1)
        print("s", s.shape, s)

        targets = torch.zeros(batch_size, 10)
        for i in range(targets.shape[0]):
            targets[i, 0] = 1
        print('targets', targets)

        loss = general_crossentropy(F.softmax(outputs, dim=1), targets)
        print("loss", loss.shape, loss)

        # Compute reference loss
        ref_loss = F.cross_entropy(outputs, torch.zeros(batch_size).long())
        print('ref_loss', ref_loss)

        # test with targets as input
        loss = general_crossentropy(targets, targets)
        print("loss_target", loss.shape, loss)

        # Compute reference loss
        ref_loss = F.cross_entropy(targets, torch.zeros(batch_size).long())
        print('ref_loss_target', ref_loss)


    def test_bock_target():
        '''Test with target defined by Sebastian Bock for tempo estimation'''
        print('TEST Bock target')
        batch_size = 16
        output_dim = 10

        outputs = torch.randn(batch_size, output_dim)
        # outputs = torch.clamp(outputs, min=0.0)
        print('outputs',outputs.shape, outputs)
        # outputs = F.softmax(outputs, dim=1)
        s = torch.sum(outputs,1)
        print("s", s.shape, s )

        targets = torch.zeros(batch_size,10)
        for i in range(targets.shape[0]):
            targets[i, 0] = 0.1
            targets[i, 1] = 0.2
            targets[i, 2] = 0.4
            targets[i, 3] = 0.2
            targets[i, 4] = 0.1
        print('targets',targets)

        loss = general_crossentropy(F.softmax(outputs, dim=1), targets)
        print("loss",loss.shape, loss)
        # Compute reference loss
        ref_loss = F.cross_entropy(outputs, torch.zeros(batch_size).long())
        print('ref_loss',ref_loss)


        # test with targets as input
        loss = general_crossentropy(targets, targets)
        print("loss_target", loss.shape, loss)

        # Compute reference loss
        ref_loss = F.cross_entropy(targets, torch.zeros(batch_size).long())
        print('ref_loss_target', ref_loss)

    def test_boeck():
        batch_size = 3
        output_dim = 11

        preds = torch.randn(batch_size, output_dim)
        print('preds', preds.shape, preds)
        label_class_idx = torch.Tensor([5, 1, 8])
        xent = XentBoeck(reduction = 'none')
        loss = xent(preds, label_class_idx)
        print('loss', loss)

    # test_onehot()
    # test_bock_target()
    test_boeck()