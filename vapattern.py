import numpy as np
import pandas as pd

import filemini as fm
from config import *

# This file contains methods performing context-aware variation feature extracting process

def _variation_feature_extraction(ipdseg: pd.Series, f_median: float = None, b_median: float = None):
    forward_derivative = (ipdseg.shift(-1) - ipdseg).dropna().values[1:]
    backward_derivative = (ipdseg.shift(1) - ipdseg).dropna().values[:-1]
    forward_derivative_abs_median = np.median(np.abs(forward_derivative)) if f_median == None else f_median
    backward_derivative_abs_median = np.median(np.abs(backward_derivative)) if b_median == None else b_median

    discretized_temp_matrix = np.zeros(shape=(len(forward_derivative), 4))
    discretized_temp_matrix[:, 0] = (forward_derivative > 0).astype(int)
    discretized_temp_matrix[:, 1] = (backward_derivative > 0).astype(int)
    discretized_temp_matrix[:, 2] = (np.abs(forward_derivative) > forward_derivative_abs_median).astype(int)
    discretized_temp_matrix[:, 3] = (np.abs(backward_derivative) > backward_derivative_abs_median).astype(int)

    binary_mask_matrix = np.array([pow(2, 3 - i) for i in range(0, 4)])
    discretized_ipdseg = (discretized_temp_matrix * binary_mask_matrix).sum(axis = 1)
    return discretized_ipdseg, forward_derivative_abs_median, backward_derivative_abs_median

def variation_feature_extraction(ipds: pd.Series, f_median: float = None, b_median: float = None, savedir: str = None, save_filename: str = None):
    # This method extracts the variation features of given traffic, outputing variation-meaningful discrete IPDs and discretization parameters(optionally)
    # The discrete IPDs and discretization parameters will be save into "savedir + save_filename" as json if the parameters of "savedir" and "save_filename" are not None.

    discretized_ipds, forward_median, backward_median = _variation_feature_extraction(ipdseg = ipds, f_median = f_median, b_median = b_median)
    result_data = {
        "discretized_ipds": list(discretized_ipds),
        "f_median": forward_median,
        "b_median": backward_median
    }
    if savedir != None:
        file = open(savedir + save_filename + "_discretized.json", 'w')
        file.write(str(result_data))
        file.close()
    return result_data

if __name__ == "__main__":
    # example usage of the methods in this file.
    trainset_range = (0, TRAINSET_SIZE)
    testset_range = (0 + TRAINSET_SIZE, 0 + TRAINSET_SIZE + TESTSET_SIZE)
    big_http_train_ipds = fm.get_raw_ipds(filedir = BIGNET_DATA_DIRPATH, filename = "http", row_range = trainset_range, window = AVERAGE_WINDOW)
    discretized_http_train_data = neighborhood_pattern_discretization(ipds = big_http_train_ipds,
                                                                      savedir = BIGNET_MANI_DIRPATH,
                                                                      save_filename = "http_train")
    trainset_f_median = discretized_http_train_data["f_median"]
    trainset_b_median = discretized_http_train_data["b_median"]

    big_http_test_ipds = fm.get_raw_ipds(filedir = BIGNET_DATA_DIRPATH, filename = "http", row_range = testset_range, value_range = None)
    discretized_http_test_data = neighborhood_pattern_discretization(ipds = big_http_test_ipds,
                                                                     f_median = trainset_f_median,
                                                                     b_median = trainset_b_median,
                                                                     savedir = BIGNET_MANI_DIRPATH,
                                                                     save_filename = "http_test")

    big_IPCTC_test_ipds = fm.get_raw_ipds(filedir = BIGNET_DATA_DIRPATH, filename = "IPCTC", row_range = testset_range, value_range = None)
    discretized_IPCTC_test_data = neighborhood_pattern_discretization(ipds = big_IPCTC_test_ipds,
                                                                      f_median = trainset_f_median,
                                                                      b_median = trainset_b_median,
                                                                      savedir = BIGNET_MANI_DIRPATH,
                                                                      save_filename = "IPCTC_test")

    big_TRCTC_test_ipds = fm.get_raw_ipds(filedir = BIGNET_DATA_DIRPATH, filename = "TRCTC", row_range = testset_range, value_range = None)
    discretized_TRCTC_test_data = neighborhood_pattern_discretization(ipds = big_TRCTC_test_ipds,
                                                                      f_median = trainset_f_median,
                                                                      b_median = trainset_b_median,
                                                                      savedir = BIGNET_MANI_DIRPATH,
                                                                      save_filename = "TRCTC_test")

    big_LNCTC_test_ipds = fm.get_raw_ipds(filedir = BIGNET_DATA_DIRPATH, filename = "LNCTC", row_range = testset_range, value_range = None)
    discretized_LNCTC_test_data = neighborhood_pattern_discretization(ipds = big_LNCTC_test_ipds,
                                                                      f_median = trainset_f_median,
                                                                      b_median = trainset_b_median,
                                                                      savedir = BIGNET_MANI_DIRPATH,
                                                                      save_filename = "LNCTC_test")

    big_Jitterbug_test_ipds = fm.get_raw_ipds(filedir = BIGNET_DATA_DIRPATH, filename = "Jitterbug", row_range = testset_range, value_range = None)
    discretized_Jitterbug_test_data = neighborhood_pattern_discretization(ipds = big_Jitterbug_test_ipds,
                                                                          f_median = trainset_f_median,
                                                                          b_median = trainset_b_median,
                                                                          savedir = BIGNET_MANI_DIRPATH,
                                                                          save_filename = "Jitterbug_test")