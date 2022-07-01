import nnrelated as nn
import detect as dt
import vapattern as va
import filemini as fm
# import datamini as dm
from config import *
from keras.models import load_model
import numpy as np
from sklearn.metrics import auc

# ***** This file includes methods conducting the experiments on GAS ****** #

def get_discretized_legit_data(filename: str = None):
    # This method gets raw continuous IPDs from file and outputs the discretized IPDs
    trainset_range = (0, TRAINSET_SIZE)
    labnet_trainset_ipds = fm.get_raw_ipds(filedir = LABNET_DATA_DIRPATH, filename = filename, row_range = trainset_range)
    discretized_labnet_trainset_data = va.variation_feature_extraction(ipds = labnet_trainset_ipds, savedir = LABNET_MANI_DIRPATH, save_filename = "http_train")
    discretized_labnet_trainset_ipds = discretized_labnet_trainset_data["discretized_ipds"]

    bignet_trainset_ipds = fm.get_raw_ipds(filedir = BIGNET_DATA_DIRPATH, filename = filename, row_range = trainset_range)
    discretized_bignet_trainset_data = va.variation_feature_extraction(ipds = bignet_trainset_ipds, savedir = BIGNET_MANI_DIRPATH, save_filename = "http_train")
    discretized_bignet_trainset_ipds = discretized_bignet_trainset_data["discretized_ipds"]
    return discretized_labnet_trainset_ipds, discretized_bignet_trainset_ipds

def train_model(discretized_labnet_trainset_ipds: np.ndarray, discretized_bignet_trainset_ipds: np.ndarray, lstm_nodes: int = None, lstm_layers: int = None):
    # This method uses legitimate discrete IPDs to train the LSTM model
    http_trainset_X, http_trainset_Y = nn.construct_sequential_prediction_dataset(raw_data = discretized_labnet_trainset_ipds, time_step_x = 1, time_step_y = TIME_STEP_Y, if_unique = False)
    model = nn.construct_LSTM_model(time_step = TIME_STEP_Y, discretized_size = 16, embedding_size = EMBEDDING_SIZE, lstm_nodes = LSTM_NODES if lstm_nodes == None else lstm_nodes)
    nn.fit_model(model = model, dataset_X = http_trainset_X, dataset_Y = http_trainset_Y, epochs = 5, savedir = LABNET_MODELS_DIRPATH, savename = "http_train")

    http_trainset_X, http_trainset_Y = nn.construct_sequential_prediction_dataset(raw_data = discretized_bignet_trainset_ipds, time_step_x = 1, time_step_y = TIME_STEP_Y, if_unique = False)
    model = nn.construct_LSTM_model(time_step = TIME_STEP_Y, discretized_size = 16, embedding_size = EMBEDDING_SIZE, lstm_nodes = LSTM_NODES if lstm_nodes == None else lstm_nodes)
    nn.fit_model(model = model, dataset_X = http_trainset_X, dataset_Y = http_trainset_Y, epochs = 5, savedir = BIGNET_MODELS_DIRPATH, savename = "http_train")

def get_AUC(valiset_losses: list, testset_losses: list):
    TPR_scores = [dt.get_detection_result_anomaly(reference_scores = valiset_losses,
                                                  test_scores = testset_losses,
                                                  threshold_p = i) for i in range(100 - FALSE_POSITIVE_RANGE[0], 100 - FALSE_POSITIVE_RANGE[1] - 1, -1)]
    FPR_scores = [i / 100 for i in range(FALSE_POSITIVE_RANGE[0], FALSE_POSITIVE_RANGE[1] + 1)]
    AUC = auc(FPR_scores, TPR_scores)
    return AUC

def sample_sensitivity_GAS(model, valiset_ipds: np.ndarray, testset_ipds: np.ndarray, f_median: float, b_median: float):
    # This method outputs the AUC scores of GAS under different sample sizes
    discretized_valiset_ipds = va.variation_feature_extraction(ipds = valiset_ipds,
                                                                      f_median=f_median,
                                                                      b_median=b_median)["discretized_ipds"]
    discretized_testset_ipds = va.variation_feature_extraction(ipds=testset_ipds,
                                                                      f_median=f_median,
                                                                      b_median=b_median)["discretized_ipds"]
    GAS_test_scores = list()
    for sample_length in SAMPLE_LENGTHS:
        valiset_losses = dt.get_testset_loss(model = model,
                                             discretized_data = discretized_valiset_ipds,
                                             sample_length = sample_length,
                                             time_step_x = 1,
                                             time_step_y = TIME_STEP_Y)
        testset_losses = dt.get_testset_loss(model=model,
                                             discretized_data = discretized_testset_ipds,
                                             sample_length = sample_length,
                                             time_step_x = 1,
                                             time_step_y = TIME_STEP_Y)
        AUC = get_AUC(valiset_losses, testset_losses)
        GAS_test_scores.append(AUC)
    return GAS_test_scores

def sample_sensitivity_evaluation():
    # This method runs sensitivity evaluation on GAS, and prints the AUC score under different sample sizes in each experiment

    # whether to retrain the model
    TRAIN_MODEL_ON = 0
    # whether to save the evaluation outcomes
    SAVE_ON = 0

    trainset_name = "http"
    if TRAIN_MODEL_ON:
        # get discretized legitimate data
        discretized_labnet_trainset_ipds, discretized_bignet_trainset_ipds = get_discretized_legit_data(filename = trainset_name)
        # train the model
        train_model(discretized_labnet_trainset_ipds = discretized_labnet_trainset_ipds, discretized_bignet_trainset_ipds = discretized_bignet_trainset_ipds)
    trainset_range = (0, TRAINSET_SIZE)
    valiset_name = "http"
    valiset_range = (TRAINSET_SIZE, TRAINSET_SIZE + TESTSET_SIZE)
    # tested channels
    testset_names = ["IPCTC", "TRCTC", "LNCTC", "Jitterbug"]
    testset_range = (0, TESTSET_SIZE)
    # get discretization parameterse
    labnet_trainset_f_median, labnet_trainset_b_median = fm.get_discretized_parameters(filedir = LABNET_MANI_DIRPATH, filename = "http_train_discretized")
    bignet_trainset_f_median, bignet_trainset_b_median = fm.get_discretized_parameters(filedir = BIGNET_MANI_DIRPATH, filename = "http_train_discretized")

    for testset_name in testset_names:
        print("Sample-sensitivity evaluation towards", testset_name)
        result_info = "Sample-sensitivity evaluation towards " + testset_name + '\n'
        # each channel we set four experiments using different groups of datasets.
        for i in range(0, 4):
            print("Exp group:", i)
            result_info += "Exp group: " + str(i) + '\n'
            model = load_model(SETTINGS[i][0] + "http_train.h5")
            f_median = labnet_trainset_f_median if i < 2 else bignet_trainset_f_median
            b_median = labnet_trainset_b_median if i < 2 else bignet_trainset_b_median
            trainset_ipds = fm.get_raw_ipds(filedir = SETTINGS[i][1], filename = "http", row_range = trainset_range)
            valiset_ipds = fm.get_raw_ipds(filedir = SETTINGS[i][1], filename = valiset_name, row_range = valiset_range)
            testset_ipds = fm.get_raw_ipds(filedir = SETTINGS[i][1], filename = testset_name + '_' + str(i % 2 + 1), row_range = testset_range)
            GAS_test_scores = sample_sensitivity_GAS(model = model, valiset_ipds = valiset_ipds, testset_ipds = testset_ipds, f_median = f_median, b_median = b_median)
            print("GAS test scores:", GAS_test_scores)
            result_info += "GAS test scores: " + str(GAS_test_scores) + '\n'
        if SAVE_ON:
            file = open(EXP_RESULT_DIRPATH + "sample_sensitivity_results.txt", 'a')
            file.write(result_info)
            file.close()


if __name__ == "__main__":
    sample_sensitivity_evaluation()








