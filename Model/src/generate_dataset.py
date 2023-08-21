import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import cv2
import torch

from submodules.UsefulTools.FileTools.PickleOperator import save_pickle
from src.data_process import CSVColumnNames


def generate_hit_datasets(df: pd.DataFrame, max_frame_id: int, img_dir: Path, target_dir: Path):
    # create a cube that contains 13 sequence frames, 7th is the HitFrame
    # one pickle is contain: [cube, process_label] the format is all tensor
    for i in range(len(df)):
        series = df.iloc[i]

        frame_id = series.at[CSVColumnNames.HitFrame]

        frame_start = frame_id - 6
        frame_end = frame_id + 6
        frame_seq_nums = np.array(range(frame_start, frame_end + 1), dtype=np.int32)
        frame_seq_nums[frame_seq_nums < 1] = 1
        frame_seq_nums[frame_seq_nums > max_frame_id] = max_frame_id

        try:
            process_label = torch.zeros(23, dtype=torch.float32)
            process_label[0 if series.at[CSVColumnNames.Hitter] == 'A' else 1] = 1.0  # Hitter: [0,1] one-hot
            process_label[1 + series.at[CSVColumnNames.RoundHead]] = 1.0  # RoundHead: [2,3] one-hot
            process_label[3 + series.at[CSVColumnNames.Backhand]] = 1.0  # Backhand: [4,5] one-hot
            process_label[5 + series.at[CSVColumnNames.BallHeight]] = 1.0  # BallHeight: [6,7] one-hot
            process_label[7 + series.at[CSVColumnNames.BallType]] = 1.0  # BallType: [8:16] one-hot
            process_label[17:23] = torch.from_numpy(
                series.loc[
                    [
                        CSVColumnNames.LandingX,  # LandingX: 17
                        CSVColumnNames.LandingY,  # LandingY: 18
                        CSVColumnNames.HitterLocationX,  # HitterLocationX: 19
                        CSVColumnNames.HitterLocationY,  # HitterLocationY: 20
                        CSVColumnNames.DefenderLocationX,  # DefenderLocationX: 21
                        CSVColumnNames.DefenderLocationY,  # DefenderLocationY: 22
                    ],
                ].to_numpy(dtype=np.float32)
            )
        except KeyError:
            os.rmdir(str(target_dir))
            break

        imgs = torch.tensor(
            np.stack([cv2.imread(str(img_dir / f'{idx}.jpg'), cv2.COLOR_BGR2RGB).transpose(2, 0, 1) for idx in frame_seq_nums])
        )

        save_pickle((imgs, process_label), str(target_dir / f'{frame_start}.pickle'))

