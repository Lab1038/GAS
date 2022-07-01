import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Flatten
import filemini as fm
from config import *

# **** This file contains methods related to the construction and utilization of neural networks **** #

def construct_sequential_prediction_dataset(raw_data: np.ndarray, time_step_x: int, time_step_y: int, if_unique: bool = False):
    # This method constructs the sequential-prediction dataset;
    # "time_step_x": IPD picking step (normally = 1)
    # "time_step_y": variation pattern length
    raw_data = np.array(raw_data)
    rolling_matrix_rows_count = int(len(raw_data) / time_step_x)
    rolling_matrix = raw_data[:rolling_matrix_rows_count * time_step_x].reshape((rolling_matrix_rows_count, time_step_x)).T
    assert rolling_matrix_rows_count > time_step_y + 1
    temp_dataset_shape = (time_step_x * (rolling_matrix_rows_count - time_step_y), time_step_y + 1)
    temp_dataset = np.zeros(shape = temp_dataset_shape)
    idx = 0
    for row in rolling_matrix:
        for i in range(len(row) - time_step_y):
            temp_dataset[idx] = row[i: i + time_step_y + 1]
            idx += 1
    if if_unique:
        temp_dataset = np.unique(temp_dataset, axis = 0)
    dataset_X = temp_dataset[:, :-1]
    dataset_Y = temp_dataset[:,-1]
    return dataset_X, dataset_Y

def construct_LSTM_model(time_step: int, discretized_size: int, embedding_size: int, lstm_nodes: int):
    # embedding layer --> LSTM layer --> dense layer
    model = Sequential()
    model.add(Embedding(input_dim = discretized_size, output_dim = embedding_size, input_length = time_step))
    model.add(LSTM(units = lstm_nodes, input_shape = (time_step, embedding_size), return_sequences = False))
    model.add(Dense(units = discretized_size, activation = "softmax"))
    model.compile(loss = "sparse_categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])
    model.summary()
    return model


def fit_model(model, dataset_X: np.ndarray, dataset_Y: np.ndarray, epochs: int, savedir: str = None, savename: str = None):
    model.fit(dataset_X, dataset_Y, epochs = epochs)
    if savedir != None:
        model.save(savedir + savename + ".h5")

if __name__ == "__main__":
    # example usage
    http_trainset = fm.get_discretized_data(filedir = LABNET_MANI_DIRPATH, filename = "http_train_discretized")
    http_trainset_X, http_trainset_Y = construct_sequential_prediction_dataset(raw_data = http_trainset, time_step_x = 1, time_step_y = TIME_STEP_Y, if_unique = False)
    model = construct_LSTM_model(time_step = TIME_STEP_Y, discretized_size = 16, embedding_size = EMBEDDING_SIZE, lstm_nodes = LSTM_NODES)
    fit_model(model = model, dataset_X = http_trainset_X, dataset_Y = http_trainset_Y, epochs = 5, savedir = LABNET_MODELS_DIRPATH, savename = "http_train_" + str(time_step_x))