import pandas as pd
import numpy as np
from config import *

# **** This file contains all file manipulation methods of our program **** #

def ipds_window_manipulate(ipds: pd.Series, window: int):
    # This method can smooth the IPD sequence by computing window mean if necessary.
    windows_count = int(len(ipds) / window)
    manipulate_matrix = ipds.values[:windows_count * window].reshape((windows_count, window))
    window_mean = np.mean(manipulate_matrix, axis=1)
    ipds = pd.Series(window_mean)
    return ipds

def get_raw_ipds(filedir: str, filename: str, row_range: tuple = None, window: int = None, value_range: tuple = (1, 99)):
    # This method reads the csv file "filedir + filename + ".csv" and return all the IPDs it contains.
    ipds = pd.read_csv(filedir + filename + ".csv")["IPDs"].dropna()
    if value_range != None:
        up = ipds.quantile(q = value_range[1] / 100)
        down = ipds.quantile(q = value_range[0] / 100)
        ipds = ipds[ipds[ipds >= down].index].reset_index(drop = True)
        ipds = ipds[ipds[ipds <= up].index].reset_index(drop = True)

    if row_range != None:
        assert row_range[0] >= 0 and row_range[1] >= 0
        if row_range[1] > len(ipds):
            row_range = (row_range[0], len(ipds))
        ipds = ipds[row_range[0]: row_range[1]]
    print("IPDs filepath:", filedir + filename + ".csv")
    print("IPDs count:", len(ipds))
    print("IPDs range:", row_range)
    # normally do not need
    if window != None:
        ipds = ipds_window_manipulate(ipds = ipds, window = window)
    return ipds

def get_discretized_parameters(filedir: str, filename: str):
    # read corresponding file and get discretization parameters.
    file = open(filedir + filename + ".json", 'r')
    data = eval(file.read())
    return data["f_median"], data["b_median"]

def get_discretized_data(filedir: str, filename: str, limit: int = None):
    # read corresponding file and get discretized IPDs.
    file = open(filedir + filename + ".json", 'r')
    data = eval(file.read())
    d_ipds = data["discretized_ipds"]
    if limit != None:
        d_ipds = d_ipds[:limit]
    print("Discretized data filepath:", filedir + filename + ".json")
    print("Discretized IPDs count:", len(d_ipds))
    return d_ipds

if __name__ == "__main__":
    pass