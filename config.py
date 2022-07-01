
# **** This file contains the experimental configuration **** #


# -------------------- paths ------------------------------ #
BIGNET_DATA_DIRPATH = "data/bignet/"
LABNET_DATA_DIRPATH = "data/labnet/"

BIGNET_MANI_DIRPATH = "manipulate/bignet/"
LABNET_MANI_DIRPATH = "manipulate/labnet/"

BIGNET_MODELS_DIRPATH = "models/bignet/"
LABNET_MODELS_DIRPATH = "models/labnet/"

EXP_RESULT_DIRPATH = "exp/"

# ----------------------- CTCs ---------------------------- #
IPDS_COUNT = 10 * 1000 * 1000
REFERENCE_DATA_DIRPATH = BIGNET_DATA_DIRPATH
IPDS_SAVE_DIRPATH = BIGNET_DATA_DIRPATH

# IPCTC
IPCTC_PARAMETERS = {
    "time_interval": [10/1000, 20/1000, 30/1000],
    "rotated_interval": 100,
    "noise": None,
    "frame_size": 100,
    "ipds_count": IPDS_COUNT,
    "save_generated_ipds_path": IPDS_SAVE_DIRPATH + "IPCTC_1.csv"
}

# TRCTC
TRCTC_PARAMETERS = {
    "ipds_count": IPDS_COUNT,
    "legit_dataset_path": REFERENCE_DATA_DIRPATH + "http.csv",
    "save_generated_ipds_path": IPDS_SAVE_DIRPATH + "TRCTC_1.csv",
    "rules": {'0':[(0, 0.45)], '1':[(0.55, 1)]},
}

# Jitterbug
Jitterbug_PARAMETERS = {
    "ipds_count": IPDS_COUNT,
    "legit_dataset_path": REFERENCE_DATA_DIRPATH + "http.csv",
    "save_generated_ipds_path": IPDS_SAVE_DIRPATH + "Jitterbug_1.csv",
    "omega": 5 / 1000, #ms
}


# LNCTC
LNCTC_PARAMETERS = {
    "ipds_count": IPDS_COUNT,
    "save_generated_ipds_path": IPDS_SAVE_DIRPATH + "LNCTC_1.csv",
    "big_delta": 10 / 1000,
    "small_delta": 5 / 1000,
    "L": 8,
    "n": 3,
    "K": 13
}

# ----------------------- others ---------------------------- #


TIME_STEP_Y = 8
EMBEDDING_SIZE = 100
LSTM_NODES = 64


# --------------------- experiments ------------------------ #
TRAINSET_SIZE = 1000 * 1000
TESTSET_SIZE = 1000 * 1000

SETTINGS = [[LABNET_MODELS_DIRPATH, LABNET_DATA_DIRPATH, 1],
            [LABNET_MODELS_DIRPATH, LABNET_DATA_DIRPATH, 2],
            [BIGNET_MODELS_DIRPATH, BIGNET_DATA_DIRPATH, 1],
            [BIGNET_MODELS_DIRPATH, BIGNET_DATA_DIRPATH, 2]]
FALSE_POSITIVE_RANGE = (0, 100)
SAMPLE_LENGTHS = range(100, 2100, 100)