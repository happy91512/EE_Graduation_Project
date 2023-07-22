import cv2
import time
import numpy as np
import pandas as pd
import PyQt6
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QMovie
from PyQt6.QtWidgets import QFileDialog, QHeaderView, QWidget
from PyQt6.QtCore import *
import sys

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
# print(sys.path)
from src.UI.UI_library import pandasModel, select_show_columns, convert_cv_qt, check_video_thread, get_time_format, cut2round
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
        self.pause = False
        
    def run(self):
        cap = cv2.VideoCapture(self.source, self.frame_target)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = 0
        while(True):
            if self.jump == True and self.frame_target != None:
                cap.set(1, self.frame_target)
                frame_count = self.frame_target
                self.jump = False  

            if not(self.pause):
                ret, self.frame = cap.read()
                if not ret:
                    time.sleep(1 / fps  * 0.81)
                    continue
                frame_count += 1
                self.slider_setup_signal.emit((frame_count, fps, length))
            try:    
                self.change_pixmap_signal.emit(self.frame)
                time.sleep(1 / fps * 0.81)
            except:continue
            
class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("badminton demo")
        self.ui_radiolist = [self.ui.groupBox.findChild(QtWidgets.QRadioButton, f"radioButton_{i}") for i in range(1, 11)]
        self.ui_giflist = [self.ui.groupBox.findChild(QtWidgets.QLabel, f"gif_{i}") for i in range(1, 11)]
        self.setup_default()
        self.setup_control()
        
    def setup_default(self):
        self.ui.slider.setDisabled(True)
        self.ui.play_pause.setDisabled(True)
        self.video_count = 0
        self.video_thread = None
        self.target_frame = 0
        self.target_frame_his = [0, 1]
        self.path_history = []
        self.temp = []
        for radio_button in self.ui_radiolist:
            radio_button.setVisible(False)
        for gif in self.ui_giflist:
            gif.setVisible(False)

    def setup_button(self):
        self.ui.slider.setDisabled(False)
        self.ui.play_pause.setDisabled(False)
        
    def setup_control(self):
        self.ui.select_file.clicked.connect(self.select_video)
        self.ui.show_csv.clicked.connect(self.open_page)
        self.ui.play_pause.clicked.connect(self.set_play_pause)
        self.ui.slider.valueChanged.connect(self.set_frame)
        self.ui_radiolist:list[QtWidgets.QRadioButton]
        for radio_button in self.ui_radiolist:
            radio_button.toggled.connect(self.check_ratio_button)
        
    def check_ratio_button(self):
        self.checklist = [wiget.isChecked() for wiget in self.ui_radiolist]
        if self.temp == self.checklist:
            return
        self.temp = self.checklist
        self.filename = self.path_history[self.checklist.index(True)]
        self.ui.show_path.setText(f"Successfully open {self.filename}!")
        self.run_video()

    def show_ratio_button(self, target_radiolist:list[QtWidgets.QRadioButton]):
        target_radiolist = [target_radiolist]
        len_limit = [16]
        self.show_gif()
        if hasattr(self, 'nw'):
            target_radiolist.append(self.nw_radiolist)
            len_limit.append(23)
        for index, radio_list in enumerate(target_radiolist):
            limit = len_limit[index]
            for i in range(self.video_count):
                radio_list[i].setVisible(True)
                if len(self.path_history[i].split('/')[-1]) > limit:
                    video_name = f"...{self.path_history[i].split('/')[-1][-limit:]}"
                else: video_name = self.path_history[i].split('/')[-1]
                radio_list[i].setText(video_name)
            if radio_list == self.ui_radiolist:
                radio_list[self.video_count - 1].setChecked(True)
    
    def show_gif(self):
        gif_movie = QMovie('./loading.gif')
        gif_movie.setScaledSize(self.ui.gif_1.size())
        for i in range(self.video_count):
            self.ui_giflist[i].setVisible(True)
            self.ui_giflist[i].setMovie(gif_movie)
        gif_movie.start()


    def select_video(self):
        self.filename, filetype = QFileDialog.getOpenFileName(self, "Open file", "./", filter = "*.avi *.mp4 *.wmv *.mov")# start path 
        if len(self.filename) == 0:
            self.ui.show_path.setText("Fail to select the video!")
            return
        elif self.filename in self.path_history:
            self.ui.show_path.setText("The video had been selected!")
            self.ui_radiolist[self.path_history.index(self.filename)].setChecked(True)
        else:
            self.path_history.append(self.filename)
            self.video_count += 1
            self.show_ratio_button(self.ui_radiolist)
            
    def run_video(self):
        check_video_thread(self.video_thread)
        self.setup_button()
        self.video_thread = VideoThread(self.filename)
        self.video_thread.change_pixmap_signal.connect(self.update_frame)
        self.video_thread.slider_setup_signal.connect(self.setup_slider)
        self.video_thread.start()
        self.ui.show_video.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

    @pyqtSlot(np.ndarray)
    def update_frame(self, cv_img):
        h = self.ui.show_video.frameGeometry().height()
        qt_img = convert_cv_qt(cv_img, h)
        round_qt_img = cut2round(qt_img)
        self.ui.show_video.setPixmap(round_qt_img)

    @pyqtSlot(tuple)
    def setup_slider(self, signal):
        (self.frame_count, fps, length) = signal
        self.ui.slider.setRange(1, length)
        self.ui.slider.setValue(self.frame_count)
        str_format = get_time_format(self.frame_count, fps)
        self.ui.video_time.setText(str_format)
    
    def set_frame(self):
        self.target_frame = self.ui.slider.value()
        if self.video_thread.pause == True:
            return
        else: 
            if self.target_frame != self.target_frame_his[-1]:
                self.target_frame_his.append(self.target_frame)
            if len(self.target_frame_his) > 4:
                self.target_frame_his = [self.target_frame, self.target_frame+1]
            if abs(self.target_frame - self.target_frame_his[-2]) > 2:
                self.video_thread.jump = True
                self.video_thread.frame_target = self.target_frame
                self.target_frame_his = [self.target_frame, self.target_frame+1]
 
    def set_play_pause(self):
        self.video_thread.pause = not(self.video_thread.pause)
        self.ui.slider.setDisabled(self.video_thread.pause)


    def open_page(self):
        self.newWindow = QtWidgets.QMainWindow()
        self.nw = Ui_new_page()
        self.nw.setupUi(self.newWindow)
        self.newWindow.show()
        self.setup_nw_control()
        self.change_csv()
        self.show_ratio_button(self.nw_radiolist)

    def setup_nw_control(self):  
        self.checkboxes = [self.nw.checkBox, self.nw.checkBox_2, self.nw.checkBox_3, self.nw.checkBox_4, self.nw.checkBox_5, self.nw.checkBox_6]
        for box in self.checkboxes:
            box.clicked.connect(self.change_csv)
        self.nw_radiolist = [self.nw.groupBox.findChild(QtWidgets.QRadioButton, f"radioButton_{i}") for i in range(1, 11)]
        for radio_button in self.nw_radiolist:
            radio_button.setVisible(False)
            radio_button.toggled.connect(self.change_csv)
        
    def check_df(self):
        self.checklist = [wiget.isChecked() for wiget in self.nw_radiolist]
        if self.temp == self.checklist:
            return
        self.temp = self.checklist
        csv_path = self.path_history[self.checklist.index(True)]


    def change_csv(self):
        checkbox_state = [box.isChecked() for box in self.checkboxes]
        df = pd.read_csv('saved_csv/00001_S2.csv').iloc[:, :len(checkbox_state)]
        show_df = select_show_columns(df, checkbox_state)  if True in checkbox_state else pd.DataFrame([['Select'],['one'],['column']]).T
        self.model = pandasModel(show_df)
        self.nw.tableView.setModel(self.model)
        header = self.nw.tableView.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

