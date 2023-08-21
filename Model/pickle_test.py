from submodules.UsefulTools.FileTools.PickleOperator import load_pickle
import cv2

# hit pickle
pickle = load_pickle('Model/Data/predict_test/00001/00001/hit/32.pickle')

print(pickle[0].shape)  # torch.Size([13, 3, 720, 1280])    #! Dataset
print(pickle[1].shape)  # torch.Size([23])  #!labal

# get5dir pickle
pickle = load_pickle('Model/Data/predict_test/00001/00001/ball_mask5_dir/36.pickle')

print(pickle.shape) # (5, 720, 1280) #? ball predict
# cv2.imwrite('test.png', pickle[0])
