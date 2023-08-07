import os, random, time
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import nn
from torchvision import transforms

from src.training import ModelPerform
from src.transforms import IterativeCustomCompose
from src.data_process import DatasetInfo, Frame13Dataset  # , get_test_dataloader
from src.evaluate.accuracy import calculate, model_acc_names
from src.net.net import BadmintonNet, BadmintonNetOperator
from src.handcraft.handcraft import get_static_handcraft_table


class TestEvaluate:
    def __init__(
        self,
        model: nn.Module,
        side: int,
        dataset: Frame13Dataset,
        test_transforms: IterativeCustomCompose,
        acc_func=calculate,
        device='cpu',
        handcraft_table_ls: List[np.ndarray] = None,
        correspond_table: np.ndarray = None,
    ):
        self.model = model
        self.side = side
        self.len_one_data = side * 2 + 1
        self.dataset = dataset
        self.test_transforms = test_transforms
        self.acc_func = acc_func
        self.device = device
        self.handcraft_table_ls = handcraft_table_ls
        self.correspond_table = correspond_table

        self.order_id = 0

    def handcraft(self, pred: torch.Tensor, video_id: int, data_id: int, start_idx: int):
        idx = int(np.where(self.correspond_table == video_id)[0])
        start_idx += data_id + self.side
        # start_idx += data_id
        # print(start_idx)
        # print(handcraft_table_ls[idx][start_idx : start_idx + self.len_one_data])
        handcraft = self.handcraft_table_ls[idx][start_idx : start_idx + self.len_one_data]
        if handcraft.shape[0] == self.len_one_data:
            pred[: self.len_one_data] += handcraft
        return pred

    def predict(self):  # , withHandCraft: bool = False):
        acc_records = np.zeros((len(self.dataset), len(model_acc_names)), dtype=np.float32)
        acc_hand_records = np.zeros((len(self.dataset), len(model_acc_names)), dtype=np.float32)
        self.model.eval()
        start_idx_list = []
        with torch.no_grad():
            for self.order_id in range(len(self.dataset)):
                data, label, hit_idxs, isHits, start_idx = self.dataset[self.order_id]
                video_id, data_id = self.dataset.dataset_paths[self.order_id].split('/')[0::2]
                video_id, data_id = map(int, (video_id, data_id[:-7]))
                data, label = data.unsqueeze(dim=0).to(self.device), label.unsqueeze(dim=0)

                batch_coordXYs = torch.stack(
                    [
                        label[:, BadmintonNetOperator.end_idx_orders[-2] :: 2],
                        label[:, BadmintonNetOperator.end_idx_orders[-2] + 1 :: 2],
                    ],
                ).permute(
                    1, 0, 2
                )  # stack like: [[relatedX, ...], [relatedY, ...]]

                data, batch_coordXYs = self.test_transforms(data, batch_coordXYs)
                batch_coordXYs = batch_coordXYs.permute(1, 0, 2)
                label[:, BadmintonNetOperator.end_idx_orders[-2] :: 2] = batch_coordXYs[0]
                label[:, BadmintonNetOperator.end_idx_orders[-2] + 1 :: 2] = batch_coordXYs[1]

                pred = self.model(data).cpu()
                # print('pred:',pred) # perdict data
                acc_records[self.order_id, :] += self.acc_func(pred, label, hit_idxs, isHits).cpu().numpy()

                # if withHandCraft:
                #     pred = self.handcraft(pred.squeeze(), video_id, data_id, start_idx).unsqueeze(dim=0)
                pred = self.handcraft(pred.squeeze(), video_id, data_id, start_idx).unsqueeze(dim=0)
                # print('hand pred:',pred)
                acc_hand_records[self.order_id, :] += self.acc_func(pred, label, hit_idxs, isHits).cpu().numpy()

                start_idx_list.append(start_idx + data_id + self.side)

        return acc_records, acc_hand_records, start_idx_list


if __name__ == '__main__':
    device = 'cpu'
    side_range = 1
    model_path = 'out/0616-1825_BadmintonNet_BS-15_AdamW1.00e-04_Side1/bestAcc-HitFrame.pt'
    test_dir = DatasetInfo.data_dir / 'test'    # 'Data/Paper_used/pre_process/frame13' / 'test'

    sizeHW = (512, 512)
    test_iter_compose = IterativeCustomCompose(
        [
            transforms.Resize(sizeHW),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
        ],
        transform_img_size=sizeHW,
        device=device,
    )

    target_dir = 'Data/Paper_used/pre_process/frame13/test'
    source_dir = 'Data/AIdea_used/pre-process'
    sub_dir = 'ball_mask5_dir'

    handcraft_table_ls, correspond_table = get_static_handcraft_table(target_dir, source_dir, sub_dir, side_range)
    print(handcraft_table_ls)
    print(correspond_table)

    test_info = DatasetInfo(data_dir=test_dir)  #! labal
    test_dataset = Frame13Dataset(side_range, dataset_info=test_info, isTrain=False)

    net = BadmintonNet(in_seq=side_range * 2 + 1).to(device)
    net = BadmintonNetOperator.load(net, model_path, device=device)

    te = TestEvaluate(
        net,
        side_range,
        test_dataset, 
        test_iter_compose,
        calculate,
        device=device,
        handcraft_table_ls=handcraft_table_ls,
        correspond_table=correspond_table,
    )

    acc_records, acc_hand_records = te.predict()
    model_perform = ModelPerform(model_acc_names, model_acc_names, loss_records=acc_records, acc_records=acc_hand_records)
    # model_perform.loss_df.to_csv('train_acc.csv')
    model_perform.acc_df.to_csv('train_hand_acc.csv')

    print(model_perform.acc_df)

