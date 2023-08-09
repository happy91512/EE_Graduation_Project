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
from src.training import ModelPerform
from src.transforms import IterativeCustomCompose
from src.data_process import DatasetInfo, Frame13Dataset  # , get_test_dataloader
from src.evaluate.accuracy import calculate, model_acc_names
from src.handcraft.handcraft import get_static_handcraft_table
from src.handcraft.TrackPoint import TrackPoint
from src.pre_process.generate_dataset import generate_hit_datasets,generate_miss_datasets
from src.handcraft.TrackNetv2_33_predict import TrackNetV2_33

PROJECT_DIR = Path(__file__).resolve().parents[0]

class TrackDebug:
    dir = Path('Data/Paper_used/pre_process')
    image_dir = Path('image')
    predict_dir = Path('predict')
    mask_dir = Path('mask')
    predict_merge_dir = Path('predict_merge')
    ball_mask5_dir = Path('ball_mask5_dir')
    HEIGHT = 288
    WIDTH = 512

    def __init__(self,data_dir, data_id) -> None:
        self.dir = data_dir / data_id
        self.image_dir = self.dir / TrackDebug.image_dir
        self.image_dir = self.dir / TrackDebug.image_dir
        self.mask_dir = self.dir / TrackDebug.mask_dir
        self.predict_dir = self.dir / TrackDebug.predict_dir
        self.predict_merge_dir = self.dir / TrackDebug.predict_merge_dir
        self.ball_mask5_dir = self.dir / TrackDebug.ball_mask5_dir

        check2create_dir(str(self.dir))
        check2create_dir(str(self.image_dir))
        # check2create_dir(str(self.predict_dir))
        # check2create_dir(str(self.mask_dir))
        # check2create_dir(str(self.predict_merge_dir))
        check2create_dir(str(self.ball_mask5_dir))

def get5dir(
        video_path:str = 'Data/predict_test/00001/00001.mp4', 
        device:str='cpu'
    ):

    # data_dir:str = 'Data/train'
    data_dir = str(Path(video_path).parents[1])
    # print(video_path.split(data_dir+'/')[-1])
    # data_dir = video_path.rstrip('/' + video_path.split('/')[-1])
    # data_dir = 'Data/z_org_val'

    DEBUG_LS = [
        'update_frame',
        # 'predict',
        # 'get_hitRangeMasks5',
    ]

    with tf.device(device):
        tNet33 = TrackNetV2_33(PROJECT_DIR/'submodules/TrackNetv2/3_in_3_out/model906_30')


        # for multiple dir
        # filenames: List[str] = get_filenames(data_dir, '*.mp4', withDirPath=False)
        # filenames.sort(reverse=True)
        filenames: List[str] =  [video_path.split(data_dir+'/')[-1]]

        rest_data = 0
        for filename in filenames:
            data_id = filename.split('/')[0]
            debugger = TrackDebug(Path(data_dir), data_id)

            # checker
            check_ls = get_filenames(str(debugger.predict_merge_dir), '*.jpg')
            if len(check_ls) != 0:
                continue
            rest_data += 1

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
                tp.update_frame(frame, isDebug='update_frame' in DEBUG_LS)
                tp.predict(isDebug='predict' in DEBUG_LS)

            masks5_ls, mask5startFrames = tp.get_hitRangeMasks5(isDebug='get_hitRangeMasks5' in DEBUG_LS)

            for mask5, mask5startFrame in zip(masks5_ls, mask5startFrames):
                save_pickle(mask5, f'{debugger.ball_mask5_dir}/{mask5startFrame}.pickle')

            cap.release()

    print(str_format(f"rest: {rest_data}", fore='y'))
    pass

def generateDataset(
    source_main_dir:Path = Path('Data/Paper_used'),
    target_main_dir:Path = Path('Data/Paper_used/pre_process/frame13'),
    image_main_dir:Path = Path('Data/AIdea_used/pre-process'),
    data_types:List[str] = ['test'], 
    source_sub_dir_name:str = 'image'
):

    for data_type in data_types:
        filenames: List[str] = sorted(get_filenames(str(source_main_dir / data_type), '*.csv', withDirPath=False), reverse=True)    #! labal

        for filename in filenames:  #! labal
            data_id = filename.split('/')[0]
            target_dir = '/'.join([data_type, data_id])

            img_dir = image_main_dir / data_id / source_sub_dir_name
            target_dir = target_main_dir / target_dir

            df = pd.read_csv(str(source_main_dir / data_type / filename))   #! labal
            frame_ids = sorted(
                map(int, [path.replace('.jpg', '') for path in get_filenames(str(img_dir), '*.jpg', withDirPath=False)])
            )
            max_frame_id = max(frame_ids)

            hit_dir = target_dir / 'hit'
            if check2create_dir(hit_dir):
                continue
            generate_hit_datasets(df, max_frame_id, img_dir, target_dir=hit_dir)    #! labal

            miss_dir = target_dir / 'miss'
            if check2create_dir(miss_dir):
                continue
            generate_miss_datasets(df, frame_ids, img_dir, target_dir=miss_dir)
    pass

def testing(
    test_dir:str = DatasetInfo.data_dir / 'test',    # 'Data/Paper_used/pre_process/frame13' / 'test'
    target_dir:str = 'Data/Paper_used/pre_process/frame13/test',
    source_dir:str = 'Data/AIdea_used/pre-process',
    sub_dir:str = 'ball_mask5_dir',
    save_csv_path:str = 'train_hand_acc.csv',
    model_path:str = 'Model/model/bestLoss-Sum.pt', 
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

    handcraft_table_ls, correspond_table = get_static_handcraft_table(target_dir, source_dir, sub_dir, side_range)
    # print(handcraft_table_ls)
    # print(correspond_table)

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

    acc_records, acc_hand_records, start_idx_list = te.predict()
    print(start_idx_list)
    model_perform = ModelPerform(model_acc_names, model_acc_names, loss_records=acc_records, acc_records=acc_hand_records)
    # model_perform.loss_df.to_csv('train_acc.csv')

    df = model_perform.acc_df.astype(int)
    for i in range(len(df.index)):
        df.loc[i, "HitFrame"] += start_idx_list[i]
        continue
    df = df.sort_values(by='HitFrame', ignore_index=True)
    df.insert(0, column="ShotSeq", value=range(1, len(df.index)+1))
    df.to_csv(save_csv_path)

    return df

def get_game_infor(
    video_path:str
):

    #? chack if they have self dir
    video_id = video_path.split('/')[-1].split('.')[0]
    data_dir, video_name = os.path.split(video_path)
    if os.path.isfile(f'{data_dir}/{video_id}/{video_id}/{video_name}'):
        video_path = f'{data_dir}/{video_id}/{video_id}/{video_name}'
    elif video_path.split('/')[-2] != video_id and video_path.split('/')[-3] != video_id :
        # mk self dir
        # dir_path, ext = os.path.splitext(video_path)
        # os.mkdir(f'{data_dir}/{video_id}')
        os.makedirs(f'{data_dir}/{video_id}/{video_id}')
        shutil.copyfile(video_path, f'{data_dir}/{video_id}/{video_id}/{video_name}')
        video_path = f'{data_dir}/{video_id}/{video_id}/{video_name}'
        if os.path.isfile(f'{data_dir}/{video_id}_S2.csv'):
            shutil.copyfile(f'{data_dir}/{video_id}_S2.csv', f'{data_dir}/{video_id}/{video_id}/{video_id}_S2.csv')

    #? set path
    predict_test = Path(video_path).parents[1]
    data_dir, video_name = os.path.split(video_path)
    save_csv_path = str(predict_test) + '_train_hand_acc.csv'

    device = '/gpu:0'
    # TrackPoint
    get5dir(
        video_path, device
    )    # get 5 dir

    # generateDataset with labal
    generateDataset(
        source_main_dir = Path(predict_test),
        target_main_dir = Path(predict_test),
        image_main_dir = Path(predict_test),
        data_types = ['.'], 
        source_sub_dir_name = 'image'
    )   # get hit dir

    # without labal
    # os.rename(f'{data_dir}/ball_mask5_dir', f'{data_dir}/hit')
    # shutil.copyfile(f'{data_dir}/ball_mask5_dir', f'{data_dir}/hit')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # testing
    df = testing(
        test_dir = Path(predict_test),    # 'Data/Paper_used/pre_process/frame13' / 'test'
        target_dir = Path(predict_test),
        source_dir = Path(predict_test),
        sub_dir = 'ball_mask5_dir',
        save_csv_path = save_csv_path,
        model_path = str(PROJECT_DIR/'Model/model/bestLoss-Sum.pt'), 
        device=device
    )   # get csv

    return df

if __name__ == '__main__':
    tt = time.time()

    video_path = PROJECT_DIR/'Data/predict_test/00001.mp4'
    # video_path = 'Data/predict_test/00001/00001/00001.mp4'
    # video_path = 'Data/predict_test/00003.mp4'
    df = get_game_infor(str(video_path))
    print(df)

    print('spend time = ', time.time() - tt)
    pass