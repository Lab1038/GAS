import math
import random
import numpy as np
import pandas as pd
from scipy.stats import gamma, exponweib, pareto, lognorm, weibull_min
from sklearn.metrics import mean_squared_error
from config import *

# ***** This file contains the implementations of four CTC algorithms ***** #

# ----------------------- IPCTC ---------------------------- #
class IpdsGeneratorIPCTC():
    def __init__(self, parameters: dict):
        self._parameters = parameters

    def _generate_covert_msgs(self, length: int):
        msgs = ''
        for i in range(length):
            msgs += str(random.randint(0, 1))
        return msgs

    def _generate_ipds(self, msgs: str, time_interval: list, rotated_interval: int, frame_size: int, noise: int):
        frame_count = int(len(msgs) / frame_size) if len(msgs) % frame_size == 0 else int(len(msgs) / frame_size) + 1
        frames = list()
        msg_count = 0
        interval_num = 0
        for i in range(frame_count):
            ipds = list()
            ipd = time_interval[interval_num % len(time_interval)] / 2
            msgs_frame = msgs[i * frame_size: -1 if i == frame_count - 1 else (i + 1) * frame_size]
            for msg in msgs_frame:
                if msg == '1':
                    ipds.append(ipd)
                    ipd = time_interval[interval_num % len(time_interval)]
                else:
                    ipd += time_interval[interval_num % len(time_interval)]
                msg_count += 1
                if msg_count == rotated_interval:
                    interval_num += 1
            frames.append(ipds)
        return frames

    def _merge_frame(self, frames: list):
        res = list()
        for frame in frames:
            res += frame
        return res

    def generate_IPCTC_ipds(self):
        # parameters: dict-like object containing time_interval, rotated_interval, noise, frame_size and msg_length
        msgs = self._generate_covert_msgs(4 * self._parameters["ipds_count"])
        frames = self._generate_ipds(msgs, self._parameters["time_interval"], self._parameters["rotated_interval"], self._parameters["frame_size"], self._parameters["noise"])
        total_ipds = self._merge_frame(frames)
        pd.DataFrame({"IPDs": total_ipds}).to_csv(self._parameters["save_generated_ipds_path"], index = False)


# ----------------------- TRCTC----------------------------- #
class IpdsGeneratorTRCTC():
    def __init__(self, parameters: dict):
        self._parameters = parameters

    def _msgs_generation(self, msgs_len: int):
        msgs = ''
        for i in range(msgs_len):
            msgs += str(random.randint(0, 1))
        return msgs

    def _ipds_load(self, ipd_filepath: str, fillna_method: str = "ffill", multiply: int = None):
        ipds = pd.read_csv(ipd_filepath)["IPDs"]
        if fillna_method != None:
            ipds = ipds.fillna(method = fillna_method)
        if multiply != None:
            ipds = ipds * multiply
        ipds = ipds[ipds[ipds > 0].index].reset_index(drop = True)
        return ipds

    def _generate_ipds(self, msgs: str, rules: list, ipds: pd.Series):
        traffic_ipds = list()
        sorted_ipds = sorted(ipds)
        for msg in msgs:
            random_bin = rules[msg][random.randint(0, len(rules[msg]) - 1)]
            traffic_ipds.append(sorted_ipds[random.randint(int(len(sorted_ipds) * random_bin[0]), int(len(sorted_ipds) * random_bin[1]) - 1)])
        return traffic_ipds

    def generate_TRCTC_ipds(self):
        msgs = self._msgs_generation(self._parameters["ipds_count"])
        legit_idps = self._ipds_load(self._parameters["legit_dataset_path"])
        TRCTC_ipds = self._generate_ipds(msgs, self._parameters["rules"], legit_idps)
        pd.DataFrame({"IPDs": TRCTC_ipds}).to_csv(self._parameters["save_generated_ipds_path"], index = False)


# ----------------------- Jitterbug ------------------------ #
class IpdsGeneratorJitterbug():
    def __init__(self, parameters: dict):
        self._parameters = parameters

    def _msgs_generation(self, msgs_len: int):
        msgs = ''
        for i in range(msgs_len):
            msgs += str(random.randint(0, 1))
        return msgs

    def _ipds_load(self, ipd_filepath: str, fillna_method: str = "ffill", multiply: int = None):
        ipds = pd.read_csv(ipd_filepath)["IPDs"]
        if fillna_method != None:
            ipds = ipds.fillna(method = fillna_method)
        if multiply != None:
            ipds = ipds * multiply
        ipds = ipds[ipds[ipds > 0].index].reset_index(drop = True)
        return list(ipds[0: self._parameters["ipds_count"]].values)

    def _generate_ipds(self, msgs: str, omega: float, ipds: list):
        if len(msgs) > len(ipds):
            times = math.ceil(len(msgs) / len(ipds))
            ipds = (ipds * times)[:len(msgs)]
        for i in range(len(msgs)):
            if msgs[i] == '0':
                ipds[i] += (omega - ipds[i] % omega) if ipds[i] % omega != 0 else 0
            else:
                ipds[i] += (omega / 2 - ipds[i] % omega) if ipds[i] % omega <= omega / 2 else (3 * omega / 2 - ipds[i] % omega)
        return ipds

    def generate_Jitterbug_ipds(self):
        msgs = self._msgs_generation(self._parameters["ipds_count"])
        legit_idps = self._ipds_load(self._parameters["legit_dataset_path"])
        Jitterbug_ipds = self._generate_ipds(msgs, self._parameters["omega"], legit_idps)
        pd.DataFrame({"IPDs": Jitterbug_ipds}).to_csv(self._parameters["save_generated_ipds_path"], index = False)

# -------------------- LNCTC ------------------------- #
class IpdsGeneratorLNCTC():
    def __init__(self, parameters: dict):
        self._parameters = parameters

    def _msgs_generation(self, msgs_len: int):
        msgs = ''
        for i in range(msgs_len):
            msgs += str(random.randint(0, 1))
        return msgs

    def _possible_comb_generation(self, n: int, K: int):
        if n == 1:
            return [(i,) for i in range(K + 1)]
        else:
            res = list()
            for i in range(K + 1):
                for subcomb in self._possible_comb_generation(n - 1, K - i):
                    res.append((i,) + subcomb)
            return res

    def _encode_msg(self, msg: str, combs: list):
        comb = combs[int(msg, 2)]
        ipds = [self._parameters["big_delta"] + i * self._parameters["small_delta"] for i in comb]
        return ipds

    def generate_LNCTC_ipds(self):
        msg_length = int((self._parameters["ipds_count"] / self._parameters["n"]) * self._parameters["L"])
        msgs = self._msgs_generation(msg_length)
        combs = self._possible_comb_generation(self._parameters["n"], self._parameters["K"])
        LNCTC_ipds = list()
        for msg_group_index in range(int(len(msgs) / self._parameters["L"])):
            msg = msgs[msg_group_index * self._parameters["L"]: (msg_group_index + 1) * self._parameters["L"]]
            LNCTC_ipds += self._encode_msg(msg, combs)
        pd.DataFrame({"IPDs": LNCTC_ipds}).to_csv(self._parameters["save_generated_ipds_path"], index = False)


if __name__ == "__main__":
    # example usage of the methods in this file.
    # generating CTC data and saving the generated data into corresponding path.
    # channel parameters and saving path are set in the dicts named xxx(channel name)_PARAMETERS in"config.py"

    TRCTC_generator = IpdsGeneratorTRCTC(TRCTC_PARAMETERS)
    TRCTC_generator.generate_TRCTC_ipds()

    Jitterbug_generator = IpdsGeneratorJitterbug(Jitterbug_PARAMETERS)
    Jitterbug_generator.generate_Jitterbug_ipds()

    IPCTC_generator = IpdsGeneratorIPCTC(IPCTC_PARAMETERS)
    IPCTC_generator.generate_IPCTC_ipds()

    LNCTC_generator = IpdsGeneratorLNCTC(LNCTC_PARAMETERS)
    LNCTC_generator.generate_LNCTC_ipds()
