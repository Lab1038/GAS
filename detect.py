import numpy as np
import pandas as pd

import vapattern as nb
import nnrelated as nn
import filemini as fm
from keras.models import load_model
import matplotlib.pyplot as plt
from config import *
from keras.backend import sparse_categorical_crossentropy
from scipy.stats import entropy, ks_2samp
from nltk import ngrams
import gzip


# **** This file includes methods used in detect process **** #

def _get_loss(model, dataset_X: np.ndarray, dataset_Y: np.ndarray):
    assert dataset_X.shape[0] == dataset_Y.shape[0]
    predicted_Y = model.predict(dataset_X)
    losses = np.array(sparse_categorical_crossentropy(target = dataset_Y, output = predicted_Y))
    return np.mean(losses)

def get_testset_loss(model, discretized_data: np.ndarray, sample_length: int, time_step_x: int, time_step_y: int):
    # This method outputs the loss of each tested sample on the legitimate model
    discretized_data = np.array(discretized_data)
    sample_count = int(len(discretized_data) / sample_length)
    samples = discretized_data[: sample_count * sample_length].reshape((sample_count, sample_length))
    losses = list()
    for row_idx in range(samples.shape[0]):
        sample = samples[row_idx]
        sample_testset_X, sample_testset_Y = nn.construct_sequential_prediction_dataset(raw_data = sample, time_step_x = time_step_x, time_step_y = time_step_y)
        loss = _get_loss(model = model, dataset_X = sample_testset_X, dataset_Y = sample_testset_Y)
        losses.append(loss)
    return np.array(losses)

def get_detection_result_anomaly(reference_scores: list, test_scores: list, threshold_p: int):
    # This method outputs the TPR (true-positive rate) and FPR (false-positive rate) of detection.
    reference_scores = np.array(reference_scores)
    test_scores = np.array(test_scores)
    threshold_up = np.percentile(reference_scores, threshold_p)
    threshold_down = np.percentile(reference_scores, 100 - threshold_p)

    legit_test_count = np.sum(test_scores < threshold_up)
    illegit_test_count = len(test_scores) - legit_test_count
    TP_rate_up = illegit_test_count / len(test_scores)

    legit_test_count = np.sum(test_scores > threshold_down)
    illegit_test_count = len(test_scores) - legit_test_count
    TP_rate_down = illegit_test_count / len(test_scores)

    return max(TP_rate_up, TP_rate_down)


