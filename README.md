# GAS
This repository is the proof-of-concept code of GAS, a covert timing channel detection approach proposed in the paper entitled "Generic and Sensitive Anomaly Detection of Network Covert Timing Channels".

# GAS Guideline 

## Repository Organization

This repository contains four folders and seven python files. The four folders are:

* `data/`  This folder contains the dataset of legitimate and covert traffic IPDs. The legitimate traffic is collected from either small-scale laboratory network or high-speed backbone links. The covert traffic is generated according to four representative covert channel algorithms, named *IPCTC*, *TRCTC*, *LNCTC* and *Jitterbug*, respectively. 
* `exp/`  The experimental results will be saved into this folder if you choose to in program settings.
* `manipulate/` The intermediate data generated in the detection process, such as the discretized IPDs, will be saved into this folder if you choose to in program settings.
* `models/` This folder temporarily saves the models trained on legitimate datasets, which will be further used to analyze traffic anomalies and detect covert channels.

The seven python files are:

* `config.py` This file contains all configuration parameters in our program. Some important parameters include:

  ```python
  # directory path of legitimate traffic datasets, either from laboratory or backbone network
  BIGNET_DATA_DIRPATH = "data/bignet/"
  LABNET_DATA_DIRPATH = "data/labnet/"
  
  # determine the size of the slide window. A value of n indicates using 8 successive IPDs to predict the following n + 1 IPD.
  TIME_STEP_Y = 8
  
  # the size of trainset and testset. A train-set larger than 500k is recommended.  
  TRAINSET_SIZE = 1000 * 1000
  TESTSET_SIZE = 1000 * 1000
  
  # the range of evaluation sample lengths
  SAMPLE_LENGTHS = range(100, 2100, 100)
  ```

   

* `CTCs.py`  This file implements four covert timing channel algorithms which can generate datasets of covert traffic IPDs according to the given parameters.

* `detect.py` This file contains methods to evaluate the anomaly degree of tested samples and calculate detection results (TPR or FPR).

* `experiment.py`  This file contains methods related to the whole experimental process, including data loading, model training, detecting executing and result printing/ saving. 

* `filemini.py`  This file provides filer-reading methods.

* `nnrelated.py`  This file contains methods related to neural network operations, such as sequential-prediction dataset constructing, model training and model fitting.

* `vapattern.py` This file implements the variation feature extracting technique. The detailed information of this technique is introduced in our paper.

## Usage

You can just run the `sample_sensitivity_evaluation` method in `experiment.py` to launch the evaluation. This method prints the AUC score of `GAS` on four experimental covert channels under different sample sizes.
