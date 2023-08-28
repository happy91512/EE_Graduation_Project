import cv2
import time
import torch
import os, shutil
import pandas as pd
import tensorflow as tf
from typing import List
from pathlib import Path
from torchvision import transforms

from submodules.UsefulTools.FileTools.WordOperator import str_format
from submodules.UsefulTools.FileTools.FileOperator import get_filenames, check2create_dir
from submodules.UsefulTools.FileTools.PickleOperator import save_pickle
from src.net.net import BadmintonNet, BadmintonNetOperator
from src.testing import TestEvaluate
from src.transforms import IterativeCustomCompose
from src.data_process import DatasetInfo, Frame13Dataset  # , get_test_dataloader
from src.accuracy import calculate, model_acc_names
from src.handcraft.handcraft import get_static_handcraft_table
from src.handcraft.TrackPoint import TrackPoint, TrackDebug
from src.generate_dataset import generate_hit_datasets
from src.handcraft.TrackNetv2_33_predict import TrackNetV2_33

PROJECT_DIR = Path(__file__).resolve().parents[0]

def get5dir(
        data_dir:str = 'Data/predict_test/00001',
        filename:str = '00001/00001.mp4',
        data_id:str = '00001',
        device:str='/gpu:0'
    ):
    DEBUG_LS = [
        'update_frame',
    ]
    with tf.device(device):
        tNet33 = TrackNetV2_33(PROJECT_DIR/'submodules/TrackNetv2/3_in_3_out/model906_30')
        debugger = TrackDebug(Path(data_dir), data_id)
        
        # checker
        check_ls = get_filenames(str(debugger.image_dir), '*.jpg')
        if not len(check_ls) == 0:
            return

        print(str_format(filename, fore='y'))

        cap = cv2.VideoCapture(f'{data_dir}/{filename}')
        FRAME = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        HIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tp = TrackPoint((HIGHT, WIDTH), FRAME, model=tNet33, debugger=debugger)
        for _ in range(FRAME):
            ret, frame = cap.read()
            if ret is False:
                tp.currentFrame += 1
                continue
            # tt = time.time()
            tp.update_frame(frame, isDebug='update_frame' in DEBUG_LS) 
            tp.predict(isDebug='predict' in DEBUG_LS)
            # print('predict time = ', time.time() - tt)
        masks5_ls, mask5startFrames = tp.get_hitRangeMasks5(isDebug='get_hitRangeMasks5' in DEBUG_LS)

        for mask5, mask5startFrame in zip(masks5_ls, mask5startFrames):
            save_pickle('.', f'{debugger.ball_mask5_dir}/{mask5startFrame}.pickle')
        cap.release()

    print(str_format(f"trackPoint done", fore='y'))
    return

def generateDataset(
    dir_path:Path = Path('Model/Data/predict_test/00001'),
    filename:str = '00001/00001_S2.csv',
    data_id:str = '00001',
    source_sub_dir_name:str = 'image'
):
    img_dir = f'{dir_path}/{data_id}/{source_sub_dir_name}'

    df = pd.read_csv(str(dir_path / filename))   #! labal
    frame_ids = sorted(
        map(int, [path.replace('.jpg', '') for path in get_filenames(str(img_dir), '*.jpg', withDirPath=False)])
    )
    max_frame_id = max(frame_ids)

    hit_dir = dir_path / data_id / 'hit'
    if check2create_dir(hit_dir): return
    generate_hit_datasets(df, max_frame_id, img_dir, target_dir=hit_dir)    #! labal
    pass

class ModelPerform:
    def __init__(
        self,
        order_names: List[str],
        records: torch.Tensor,
    ) -> None:
        self.df = pd.DataFrame(self.convert(records, order_names))

    @staticmethod
    def convert(records: torch.Tensor, order_names: List[str]):
        return {name: records[:, i].tolist() for i, name in enumerate(order_names)}

def testing(
    dir_path:str = 'Data/Paper_used/pre_process/frame13/test',
    sub_dir:str = 'ball_mask5_dir',
    model_path:str = 'model/bestLoss-Sum.pt', 
    device:str = 'cpu'
):
    side_range = 1

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

    handcraft_table_ls, correspond_table = get_static_handcraft_table(dir_path, dir_path, sub_dir, side_range)
    print(handcraft_table_ls)

    test_info = DatasetInfo(data_dir=dir_path)  #! labal
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

    acc_hand_records, start_idx_list = te.predict()
    # print(start_idx_list)
    model_perform = ModelPerform(order_names=model_acc_names, records=acc_hand_records)
    # model_perform.loss_df.to_csv('train_acc.csv')

    df = model_perform.df.astype(int)
    for i in range(len(df.index)):
        df.loc[i, "HitFrame"] += start_idx_list[i]
        if df.loc[i, "Hitter"]==0:  df.loc[i, "Hitter"] = 'A'
        else: df.loc[i, "Hitter"] = 'B'
        df.loc[i, "RoundHead"] += 1
        df.loc[i, "Backhand"] += 1
        df.loc[i, "BallHeight"] += 1
        continue
    df = df.sort_values(by='HitFrame', ignore_index=True)
    df.insert(0, column="ShotSeq", value=range(1, len(df.index)+1))

    cols = df.columns.to_list()
    cols.insert(13, cols.pop(cols.index('BallType')))
    df = df[cols]

    return df

def get_game_infor(
    video_path:str
):

    #? chack if they have self dir
    video_path_split = video_path.split('/')
    video_id = video_path_split[-1].split('.')[0]
    data_dir, video_name = os.path.split(video_path)
    if os.path.isfile(f'{data_dir}/{video_id}/{video_id}/{video_name}'):
        video_path = f'{data_dir}/{video_id}/{video_id}/{video_name}'
        # data_dir, video_name = os.path.split(video_path)
    elif video_path_split[-2] != video_id and video_path_split[-3] != video_id :
        # mk self dir
        os.makedirs(f'{data_dir}/{video_id}/{video_id}')
        shutil.copyfile(video_path, f'{data_dir}/{video_id}/{video_id}/{video_name}')
        video_path = f'{data_dir}/{video_id}/{video_id}/{video_name}'
        if os.path.isfile(f'{data_dir}/{video_id}_S2.csv'):
            shutil.copyfile(f'{data_dir}/{video_id}_S2.csv', f'{data_dir}/{video_id}/{video_id}/{video_id}_S2.csv')

    #? set path
    predict_test = Path(video_path).parents[1]


    # TrackPoint
    get5dir(
        data_dir = f'{data_dir}/{video_id}',
        filename = f'{video_id}/{video_path_split[-1]}',
        data_id = video_id,
        device ='/gpu:0'
    )    # get 5 dir


    # generateDataset with labal
    generateDataset(
        dir_path = predict_test,
        filename = f'{video_id}/{video_id}_S2.csv',
        data_id = video_id,
        source_sub_dir_name = 'image'
    )   # get hit dir


    # testing
    df = testing(
        dir_path = predict_test,
        sub_dir = 'ball_mask5_dir',
        model_path = str(PROJECT_DIR/'model/bestLoss-Sum.pt'), 
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    )   # get csv

    return df

if __name__ == '__main__':
    tt = time.time()

    print(PROJECT_DIR) # PROJECT_DIR = .../Model/

    video_path = 'Model/Data/predict_test/00001.mp4'
    save_csv_path = 'Model/Data/predict_test/00001_train_hand_acc.csv'

    df = get_game_infor(str(video_path))
    df.to_csv(save_csv_path)
    print(df)

    print('spend time = ', time.time() - tt)
    pass