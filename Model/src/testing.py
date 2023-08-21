from typing import List
import numpy as np
import torch
from torch import nn

from src.transforms import IterativeCustomCompose
from src.data_process import DatasetInfo, Frame13Dataset  # , get_test_dataloader
from src.accuracy import calculate, model_acc_names
from src.net.net import BadmintonNet, BadmintonNetOperator

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
        handcraft = self.handcraft_table_ls[idx][start_idx : start_idx + self.len_one_data]
        if handcraft.shape[0] == self.len_one_data:
            pred[: self.len_one_data] += handcraft
        return pred

    def predict(self):  # , withHandCraft: bool = False):
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
                pred = self.handcraft(pred.squeeze(), video_id, data_id, start_idx).unsqueeze(dim=0)
                acc_hand_records[self.order_id, :] += self.acc_func(pred, label, hit_idxs, isHits).cpu().numpy()
                start_idx_list.append(start_idx + data_id + self.side)

        return acc_hand_records, start_idx_list
