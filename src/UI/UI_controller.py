import cv2
import time
import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QFileDialog, QHeaderView
from PyQt6.QtCore import *
import sys

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
# print(sys.path)
from src.UI.UI_library import pandasModel, select_show_columns, convert_cv_qt
from src.UI.UI import Ui_MainWindow
from src.UI.csv_page import Ui_new_page

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    slider_setup_signal = pyqtSignal(tuple)
    def __init__(self, source, frame_target = None, jump = False):
        super().__init__()
        self.source = source
        self.frame_target = frame_target
        self.jump = jump
        
    def run(self):
        cap = cv2.VideoCapture(self.source, self.frame_target)
        frame_count = 0
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        while(True):
            if self.jump == True:
                cap.set(1, self.frame_target)
                frame_count = self.frame_target
                self.jump = False  
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                self.slider_setup_signal.emit((frame_count, fps, length))
                self.change_pixmap_signal.emit(frame)
                # cv2.waitKey(25)
                time.sleep(1 / fps * 0.81)
            

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("badminton demo")
        self.setup_control()
        self.ui.slider.setDisabled(True)
        self.target_frame = 0
        self.target_frame_his = [0, 1]
        
    def setup_control(self):
        self.ui.select_file.clicked.connect(self.open_video)
        self.ui.show_csv.clicked.connect(self.open_page)
        self.ui.slider.valueChanged.connect(self.set_frame)
        
    def setup_nw_control(self):
        self.checkboxes = [self.nw.checkBox, self.nw.checkBox_2, self.nw.checkBox_3, self.nw.checkBox_4, self.nw.checkBox_5, self.nw.checkBox_6]
        for box in self.checkboxes:
            box.clicked.connect(self.change_csv)
           
    def open_video(self):
        self.filename, filetype = QFileDialog.getOpenFileName(self, "Open file", "./", filter = "*.avi *.mp4 *.wmv *.mov")# start path 
        if len(self.filename) == 0:
            self.ui.show_path.setText("Select a target video...")
            return
        else:
            self.ui.show_path.setText(self.filename)
        self.check_video_thread()
        self.video_thread = VideoThread(self.filename)
        self.video_thread.change_pixmap_signal.connect(self.update_frame)
        self.video_thread.slider_setup_signal.connect(self.setup_slider)
        self.video_thread.start()
        self.ui.slider.setDisabled(False)
        self.ui.show_video.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

    @pyqtSlot(np.ndarray)
    def update_frame(self, cv_img):
        h = self.ui.show_video.frameGeometry().height()
        qt_img = convert_cv_qt(cv_img, h)
        self.ui.show_video.setPixmap(qt_img)

    def check_video_thread(self):
        try: 
            if self.video_thread.isRunning():
                self.video_thread.terminate()
                print('stop thread')
            elif self.video_thread.isFinished():
                self.ui.show_video.clear()
                self.ui.show_video.setText('Wait for new videos...')
        except:pass

    @pyqtSlot(tuple)
    def setup_slider(self, signal):
        (self.frame_count, fps, length) = signal
        self.ui.slider.setRange(1, length)
        self.ui.slider.setValue(self.frame_count)
        secs = int(self.frame_count/fps)
        mins = int(secs / 60)
        secs = secs % 60
        self.ui.video_time.setText(f"{mins:02d}:{secs:02d}")
    
    def set_frame(self):
        self.target_frame = self.ui.slider.value()
        if self.target_frame != self.target_frame_his[-1]:
            self.target_frame_his.append(self.target_frame)
        if len(self.target_frame_his) > 4:
            self.target_frame_his = [self.target_frame, self.target_frame+1]
        if abs(self.target_frame - self.target_frame_his[-2]) > 2:
            self.video_thread.jump = True
            self.video_thread.frame_target = self.target_frame
            self.target_frame_his = [self.target_frame, self.target_frame+1]
 
    def open_page(self):
        self.newWindow = QtWidgets.QMainWindow()
        self.nw = Ui_new_page()
        self.nw.setupUi(self.newWindow)
        self.newWindow.show()
        self.setup_nw_control()
        self.change_csv()

    def change_csv(self):
        check_state = [box.isChecked() for box in self.checkboxes]
        df = pd.read_csv('./saved_csv/00001_S2.csv').iloc[:, :len(check_state)]
        show_df = select_show_columns(df, check_state)  if True in check_state else pd.DataFrame([['Select'],['one'],['column']]).T
        self.model = pandasModel(show_df)
        self.nw.tableView.setModel(self.model)
        header = self.nw.tableView.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

