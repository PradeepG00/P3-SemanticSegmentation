# import logging
from typing import List
import subprocess

import pandas
import torch.cuda


def get_gpu_memory_map() -> dict:
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]

    gpu_memory_map = {id: {"MB": gpu_memory[id], "GB": gpu_memory[id] / 1000} for id in range(len(gpu_memory))}
    return gpu_memory_map


def get_available_gpus(memory_threshold: float = 0.0, metric: str = "mb") -> List:
    """Get all the available GPUs using less memory than a specified threshold

    :param memory_threshold: maximum memory usage threshold to reject
    :param metric: GB or MB
    :return: List
    """
    m = metric.upper()
    assert m in ["GB", "MB"]
    print("GPUs using less than {} {}".format(memory_threshold, m))
    # logging.debug("GPUs using less than {}{}".format(memory_threshold, m)) # DEBUG

    gpu_mem_usage = get_gpu_memory_map()
    # print(gpu_mem_usage)    # DEBUG
    ids = []
    for gpu_id, metric2mem_dict in gpu_mem_usage.items():
        if metric2mem_dict[m] < memory_threshold:
            ids.append(gpu_id)
    return ids


def get_gpu_stats() -> pandas.DataFrame:
    """Get statistics of all GPUs in a DataFrame

    :return:
    """
    gpu_dict = {
        "id": [],
        "total": [],
        "reserve": [],
        "usage": [],
        "free": []
    }
    for i in range(9):
        try:
            print(torch.cuda.memory_stats(i))
            t = torch.cuda.get_device_properties(i).total_memory
            r = torch.cuda.memory_reserved(i) / 1024*3
            a = torch.cuda.memory_allocated(i) / 1024*3
            f = r - a  # free inside reserved
            gpu_dict["id"].append(i)
            gpu_dict["reserve"].append(r)
            gpu_dict["total"].append(t)
            gpu_dict["usage"].append(a)
            gpu_dict["free"].append(f)
            print(torch.cuda.get_device_name(i))
        except Exception as e:
            pass
    import pandas as pd
    return pd.DataFrame(gpu_dict)


# def get_gpu_stats():
#     print(get_available_gpus(90000, "mb")) # DEBUG
#     torch.cuda.get_device_properties()

if __name__ == "__main__":
    gpu_mem_usage = get_gpu_memory_map()
    print(gpu_mem_usage)
    print(get_available_gpus(10000, "mb"))
    print(get_gpu_stats())
    print(torch.cuda.device_count()) # returns 1 in my case