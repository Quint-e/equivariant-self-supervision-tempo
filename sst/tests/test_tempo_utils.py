# -*- coding: utf-8 -*-
"""
@author: Elio Quinton
python3.8

Test augmentation layers

"""
import numpy as np
import torch
import torch.nn.functional as F

from sst.utils.tempo_utils import tempo_to_onehot, onehot_to_tempo, tempo_to_class_index, class_index_to_tempo, class_index_to_onehot


class TestTempoToOnehot:
    def test_onehot_tensors(self):
        '''Test that the one-hot tensors have the right shape and values, given some tempi'''
        tempi = torch.Tensor([123,254,52,78,87.3,174.8])
        tempi = torch.unsqueeze(tempi,1) # Add "feature" dimension - for consistency with vector outputs that would have (batch_size, n_features) shape.
        tempo_range = (0,300)
        one_hots = tempo_to_onehot(tempi,tempo_range,tempo_widening_win=None)

        assert one_hots.shape == (tempi.shape[0],300), "One-hot tensors have unexpected shape {}".format(one_hots.shape)
        # Check that we have exactly one 1 per one-hot vector.
        assert torch.equal(torch.sum(one_hots,dim=1), torch.ones(tempi.shape[0])), "One-hot tensors do not sum to 1"
        # Check that the ones are at the right indexes.
        assert torch.equal(torch.argmax(one_hots,dim=1),torch.round(torch.squeeze(tempi)).long())

    def test_onehot_tensors_outofdomain(self):
        '''Test that the one-hot tensors have the right shape and values, given some out-of domain tempi'''
        tempi = torch.Tensor([300, 535, -12.3, 352.9])
        tempi = torch.unsqueeze(tempi, 1)  # Add "feature" dimension - for consistency with vector outputs that would have (batch_size, n_features) shape.
        tempo_range = (0, 300)
        one_hots = tempo_to_onehot(tempi, tempo_range, tempo_widening_win=None)

        assert one_hots.shape == (tempi.shape[0], 300), "One-hot tensors have unexpected shape {}".format(
            one_hots.shape)
        # Check that we have exactly one 1 per one-hot vector.
        assert torch.equal(torch.sum(one_hots, dim=1), torch.ones(tempi.shape[0])), "One-hot tensors do not sum to 1"
        # Check that the 1s are at the non-tempo index (0).
        assert torch.equal(torch.argmax(one_hots,dim=1),torch.zeros(tempi.shape[0]).long()), "Unexpected one-hot tensor values for out of domain tempi"

    def test_onehot_widened(self):
        '''Test that the one-hot tensors have the right shape and values, given some tempi, and applying one-hot widening'''
        tempi = torch.Tensor([123,254,52,78,87.3,174.8])
        tempi = torch.unsqueeze(tempi, 1)  # Add "feature" dimension - for consistency with vector outputs that would have (batch_size, n_features) shape.
        tempo_range = (0,300)
        widening_win = torch.Tensor([0.25,0.5,1.0,0.5,0.25])
        one_hots = tempo_to_onehot(tempi,tempo_range,tempo_widening_win=widening_win)

        assert one_hots.shape == (tempi.shape[0],300), "One-hot tensors have unexpected shape {}".format(one_hots.shape)
        # Check that widened "one-hot" vectors sum to 1.
        assert torch.equal(torch.sum(one_hots,dim=1), torch.ones(tempi.shape[0])), "One-hot tensors do not sum to 1"
        # Check that the max values are at the right indexes.
        assert torch.equal(torch.argmax(one_hots,dim=1),torch.round(torch.squeeze(tempi)).long())
        # Check that the widening pattern is correct
        widening_win_norm = F.normalize(widening_win,p=1,dim=0)
        N = widening_win.shape[0]//2
        for i, t in enumerate(tempi):
            pattern = one_hots[i, int(round(t.item()-N)):int(round(t.item()+N))+1]
            assert torch.equal(widening_win_norm,pattern), "Unexpected one-hot Widening pattern with tempo {}".format(t.item())


class TestOnehotToTempo:
    def test_tempo(self):
        tempi = torch.Tensor([145, 222, 66, 0, 53, 99, 74])
        tempi = torch.unsqueeze(tempi, 1)  # Add "feature" dimension - for consistency with vector outputs that would have (batch_size, n_features) shape.
        tempo_range = (0, 300)
        one_hot = torch.zeros(tempi.shape[0],300)
        for i, t in enumerate(tempi):
            one_hot[i,int(t.item())] = 1
        tempi_pred = onehot_to_tempo(one_hot,tempo_range)
        assert torch.equal(tempi_pred,tempi), "Tempo output does not match one-hot input."


class TestTempoToClassIndex:
    def test_class_index(self):
        tempi = torch.Tensor([123, 254, 52, 78, 87.3, 174.8])
        tempi = torch.unsqueeze(tempi, 1)  # Add "feature" dimension - for consistency with vector outputs that would have (batch_size, n_features) shape.
        tempo_range = (0, 300)
        class_indexes = tempo_to_class_index(tempi,tempo_range)
        assert torch.equal(torch.round(torch.squeeze(tempi)), class_indexes)
        # Same test with different tempo range
        tempo_range = (30, 300)
        class_indexes = tempo_to_class_index(tempi, tempo_range)
        assert torch.equal(torch.round(torch.squeeze(tempi) - tempo_range[0]), class_indexes)


class TestClassIndexToTempo:
    def test_tempo(self):
        class_indexes = torch.Tensor([155,91,42,230]).long()
        tempo_range = (20,300)
        tempi = class_index_to_tempo(class_indexes,tempo_range)
        assert torch.equal(torch.squeeze(tempi), class_indexes+tempo_range[0])


class TestClassIndexToOnehot:
    def test_onehot(self):
        class_indexes = torch.Tensor([132,65,72,267,88]).long()
        tempo_range = (0,300)
        one_hot = class_index_to_onehot(class_indexes,tempo_range)
        assert one_hot.shape == (class_indexes.shape[0],tempo_range[1]-tempo_range[0])
        assert torch.equal(torch.argmax(one_hot,dim=1),class_indexes)


class TestCircular:
    def test_tempo_onehot(self):
        tempo_range = (0,300)
        tempi = torch.Tensor([123, 254, 52, 78, 87.3, 174.8])
        tempi = torch.unsqueeze(tempi, 1)
        one_hot = tempo_to_onehot(tempi,tempo_range)
        tempi_out = onehot_to_tempo(one_hot,tempo_range)
        assert torch.equal(tempi_out,torch.round(tempi))

    def test_onehot_tempo(self):
        tempo_range = (0, 300)
        tempi = torch.Tensor([123, 254, 52, 78, 87.3, 174.8])
        tempi = torch.unsqueeze(tempi, 1)
        one_hot = torch.zeros(tempi.shape[0], np.diff(tempo_range)[0])
        for i, t in enumerate(tempi):
            one_hot[i,int(round(t.item()))] = 1
        tempi_out = onehot_to_tempo(one_hot, tempo_range)
        onehot_out = tempo_to_onehot(tempi_out, tempo_range)
        assert torch.equal(onehot_out, one_hot)


if __name__ == "__main__":
    test_tempo_to_onehot = TestTempoToOnehot()
    test_tempo_to_onehot.test_onehot_tensors()
    test_tempo_to_onehot.test_onehot_tensors_outofdomain()
    test_tempo_to_onehot.test_onehot_widened()

    test_onehot_to_tempo = TestOnehotToTempo()
    test_onehot_to_tempo.test_tempo()

    test_tempo_to_class_index = TestTempoToClassIndex()
    test_tempo_to_class_index.test_class_index()

    test_class_index_to_tempo = TestClassIndexToTempo()
    test_class_index_to_tempo.test_tempo()

    test_class_index_to_onehot = TestClassIndexToOnehot()
    test_class_index_to_onehot.test_onehot()

    test_circular = TestCircular()
    test_circular.test_tempo_onehot()
    test_circular.test_onehot_tempo()