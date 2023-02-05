# -*- coding: utf-8 -*-
"""
@author: Elio Quinton
python3.8
"""

import numpy as np
import mir_eval


def accuracy1(tempo_pred: float, tempo_true: float, tol: float = 0.04) -> bool:
    '''Returns True if the tempo_pred is within ± tol of tempo_true.'''
    if tempo_true*(1-tol) <= tempo_pred <= tempo_true*(1+tol):
        return True
    else:
        return False

def accuracy2(tempo_pred: float, tempo_true: float, tol: float = 0.04, multiples = (1, 2, 3, 0.5, 0.33)) -> bool:
    '''Returns True if the tempo_pred is within ± tol of tempo_true or its double, triple, half or third (or any other multiples provided as input).'''
    for mul in multiples:
        if accuracy1(tempo_pred, tempo_true*mul, tol=tol) is True:
            return True
    return False

def tempo_eval_basic(reference_tempi_tuple, estimated_tempi_tuple, tol=0.08):
    '''Computes basic tempo metrics using mir_eval.
    Parameters:
        - reference_tempi: (tuple): (tempo1, tempo2, weight), where tempo0 < tempo1 and weight is the relative weight.
        - estimated_tempi: (tuple): (tempo1, tempo2, weight), where tempo0 < tempo1 and weight is the relative weight.
        - tol: (float): Tolerance. Defaults to 8% of reference value.
    Returns:
        - p_score: (float)
        - one_correct: (bool)
        - both_correct: (bool)
        '''
    # Convert input data into format expected by mir_eval
    reference_tempi = np.array([reference_tempi_tuple[0],reference_tempi_tuple[1]])
    reference_weight = reference_tempi_tuple[2]
    estimated_tempi = np.array([estimated_tempi_tuple[0],estimated_tempi_tuple[1]])
    # Compute metrics using mir-eval
    p_score, one_correct, both_correct = mir_eval.tempo.detection(reference_tempi,reference_weight,estimated_tempi,tol=tol)
    return p_score, one_correct, both_correct


def tempo_eval_basic_batch(reference_tempi_list, estimated_tempi_list, tol=0.08):
    '''Computes basic tempo metrics using mir_eval for a batch of examples.
    Parameters:
        - reference_tempi: (list): list of tuple (tempo1, tempo2, weight), where tempo0 < tempo1 and weight is the relative weight.
        - estimated_tempi: (list): list of tuple (tempo1, tempo2, weight), where tempo0 < tempo1 and weight is the relative weight.
        - tol: (float): Tolerance. Defaults to 8% of reference value.
    Returns:
        - p_score: (list)
        - one_correct: (list)
        - both_correct: (list)
        '''

    batch_metrics = []
    for i, ref in enumerate(reference_tempi_list):
        metrics = tempo_eval_basic(ref,estimated_tempi_list[i],tol=tol)
        batch_metrics.append(metrics)
    # "unzip" the batch metrics into separate lists
    batch_metrics = list(zip(*batch_metrics))
    return batch_metrics[0], batch_metrics[1], batch_metrics[2]


