import os
from typing import List, Tuple

import numpy as np
import torch

from submodules.UsefulTools.FileTools.FileOperator import get_filenames


def get_static_handcraft_table(target_dir: str, data_dir: str, sub_dir: str, side: int) -> Tuple[List[np.ndarray], np.ndarray]:
    data_ids = sorted(os.listdir(target_dir))

    video_startIDs_ls = [
        sorted(map(lambda x: int(x[:-7]), get_filenames(f'{data_dir}/{data_id}/{sub_dir}', '*.pickle', withDirPath=False))) # Data/AIdea_used/pre-process/{id}/ball_mask5_dir/***.pickle
        for data_id in data_ids
    ]

    # Data/Paper_used/test
    handcraft_table_ls = []
    hand_add = 1 / (side * 2 + 1)
    hand_sub = hand_add
    side = 2  # * because vector method's pickle file is frame5

    for video_startIDs in video_startIDs_ls:
        idxs = np.array(video_startIDs) + side
        handcraft_table = torch.zeros(idxs.max() + side * 2, dtype=torch.float32)

        handcraft_table[idxs] += hand_add
        for s in range(1, side + 1):
            handcraft_table[idxs + s] += hand_add
            handcraft_table[idxs - s] += hand_add

        for idx in idxs:
            handcraft_table[idx - side * 2 : idx + side * 2] -= hand_sub
            handcraft_table[idx - side * 1 : idx + side * 1] += hand_sub
        handcraft_table_ls.append(handcraft_table)

    correspond_table = np.array([*map(int, data_ids)], dtype=np.int16)

    # print(video_startIDs_ls)
    return handcraft_table_ls, correspond_table