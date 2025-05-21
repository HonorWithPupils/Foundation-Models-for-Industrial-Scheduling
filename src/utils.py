import torch
from torch.nn import functional as F

from typing import Union


def loadFJSP(file_path: str):
    """load FJSP problem from file

    Args:
        file_path (str): path to file

    Returns:
        Dict: info of FJSP
    """

    with open(file_path, "r") as file:
        lines = file.readlines()

    # load first line
    first_line = lines[0].strip().split()
    n = int(first_line[0])
    m = int(first_line[1])
    UB = float(first_line[-1]) / 100.0

    total_operations = sum(int(line.split()[0]) for line in lines[1:])

    # initialize the FJSPinfo dictionary
    FJSPinfo = {
        "n": n,
        "m": m,
        "UB": UB,
        "o": {
            "machine": [[0] * m for _ in range(total_operations)],
            "job": [0] * total_operations,
            "time": [[0] * m for _ in range(total_operations)],
            "order": [],
        },
    }

    current_operation = 0
    for job_index, line in enumerate(lines[1:], start=0):
        data = list(map(int, line.strip().split()))
        num_operations = data[0]
        operation_order = list(range(num_operations))

        FJSPinfo["o"]["order"].extend(operation_order)

        index = 1
        for operation_index in range(num_operations):
            k = data[index]
            FJSPinfo["o"]["job"][current_operation] = job_index
            index += 1

            for _ in range(k):
                machine_id = data[index] - 1
                processing_time = data[index + 1]

                FJSPinfo["o"]["machine"][current_operation][machine_id] = 1
                FJSPinfo["o"]["time"][current_operation][machine_id] = processing_time

                index += 2

            current_operation += 1

    FJSPinfo["o"] = {k: torch.tensor(v).unsqueeze(0) for k, v in FJSPinfo["o"].items()}

    return FJSPinfo


def load_from_multiGPU(path):
    state_dict = torch.load(path, weights_only=True)
    new_state_dict = {}
    for key, value in state_dict.items():
        if "module." in key:
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict
