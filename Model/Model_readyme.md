!!!Model Notes!!!\
Please download the senior's model unzip and put it in model/\
link: https://drive.google.com/file/d/1bJWtJiURDk5fFAOekuYgftlpK39_j3TI/view?usp=sharing\
path is 'Model/model/bestLoss-Sum.pt'

and download the TrackNetv2's model906_30 and put it under submodules/...\
link: https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2/blob/master/3_in_3_out/model906_30\
path is 'Model/submodules/TrackNetv2/3_in_3_out/model906_30'

example input:\
    video_path = 'Data/predict_test/00001.mp4'\
output:\
    dataframe = get_game_infor(video_path)

Data Structure:\
    Data/predict_test/00001.mp4\
    Data/predict_test/00001_S2.csv\
!!!please make sure the 00001_S2.csv doesn't contain space!!!

change into:\
    Data/predict_test/00001.mp4\
    Data/predict_test/00001_S2.csv\
    and\
    Data/predict_test/00001/00001/00001.mp4\
    Data/predict_test/00001/00001/00001_S2.csv\
    Data/predict_test/00001/00001/...