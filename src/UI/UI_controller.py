from typing import Any
import cv2
import time
import numpy as np
import pandas as pd
import PyQt6

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QMovie
from PyQt6.QtWidgets import QFileDialog, QHeaderView, QWidget
from PyQt6.QtCore import pyqtSignal, QThread, pyqtSlot
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

matplotlib.use("Qt5Agg")
import sys

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.UI.UI_library import pandasModel, select_show_columns, convert_cv_qt, check_video_thread, get_time_format, cut2round, plt_figure2array
from src.UI.UI import Ui_MainWindow
from src.UI.csv_page import Ui_new_page
from src.data import analyze


class DetectThread(QThread):
    dataframe_signal = pyqtSignal(tuple)
    def __init__(self, filename, number):
        super().__init__()
        self.number = number
    def run(self):
        pass
        cv2.waitKey(6000)
        self.dataframe_signal.emit((pd.DataFrame([1, 2, 3]), self.number))
        return 

    
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    slider_setup_signal = pyqtSignal(tuple)
    def __init__(self, source, frame_target = None, jump = False, stop = False):
        super().__init__()
        self.source = source
        self.frame_target = frame_target
        self.jump = jump
        self.pause = False
        self.stop = stop
        
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
            if self.stop:
                return
    def cancel(self):
        self.requestInterruption() 
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
        self.figA, self.figB = plt.figure(), plt.figure()
        self.analyze_history = [False] * 10
        self.detect_done = [bool(ele) for ele in self.analyze_history]
        self.detect_thread = [None] * 10
        self.df = pd.read_csv('saved_csv/00001_S2.csv')
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
        if bool(self.analyze_history[self.checklist.index(True)]) == True:
            self.ui.show_analysis.setText(self.analyze_history[self.checklist.index(True)][0])
        self.run_video()
        
    def show_ratio_button(self, target_radiolist:list[QtWidgets.QRadioButton]):
        target_radiolist = [target_radiolist]
        len_limit = [16]
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
        gif_loading = QMovie('./loading.gif')
        gif_loading.setScaledSize(self.ui.gif_1.size())
        gif_done = QMovie('./done.gif')
        gif_done.setScaledSize(self.ui.gif_1.size())
        for i in range(self.video_count):
            self.ui_giflist[i].setVisible(True)
            self.ui_giflist[i].setMovie(gif_loading if self.detect_done[i] == False else gif_done)
        gif_loading.start()
        gif_done.start()

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
            self.run_detection()
            self.show_ratio_button(self.ui_radiolist)

    def run_detection(self):
        self.show_gif()
        number = self.path_history.index(self.filename)
        self.detect_thread[number] = DetectThread(self.filename, number)
        self.detect_thread[number].dataframe_signal.connect(self.check_detect_state)
        self.detect_thread[number].start()
        
    @pyqtSlot(tuple)
    def check_detect_state(self, df_tuple):
        df, num = df_tuple
        #! need to be modified
        df = self.df
        self.str_list, self.figA, self.figB = analyze(df)
        self.analyze_history[num] = [str(num)+'\n'+self.str_list, df, self.figA, self.figB]
        self.detect_done = [bool(ele) for ele in self.analyze_history]
        if hasattr(self, 'nw'):
            for i, e in enumerate(self.detect_done):
                self.nw_radiolist[i].setDisabled(not e)
             
        self.ui.show_analysis.setText(self.analyze_history[num][0])
        self.show_gif()

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
        self.setup_nw_default()
        self.show_ratio_button(self.nw_radiolist)

    def setup_nw_control(self):  
        self.checkboxes = [self.nw.checkBox, self.nw.checkBox_2, self.nw.checkBox_3, self.nw.checkBox_4, self.nw.checkBox_5, self.nw.checkBox_6]
        for box in self.checkboxes:
            box.clicked.connect(self.change_csv)
        self.nw_radiolist = [self.nw.groupBox.findChild(QtWidgets.QRadioButton, f"radioButton_{i}") for i in range(1, 11)]
        for radio_button in self.nw_radiolist:
            radio_button.setVisible(False)
            radio_button.toggled.connect(self.change_csv)
            radio_button.toggled.connect(self.update_figure)

    def setup_nw_default(self):
        for i, e in enumerate(self.detect_done):
                self.nw_radiolist[i].setDisabled(not e)

    def check_df(self):
        self.checklist = [wiget.isChecked() for wiget in self.nw_radiolist]
        if self.temp == self.checklist:
            return
        self.temp = self.checklist
        # csv_path = self.path_history[self.checklist.index(True)]

    def change_csv(self):
        checkbox_state = [box.isChecked() for box in self.checkboxes]
        # self.df = self.analyze_history[self.checklist.index(True)][1]
        show_df = pd.read_csv('saved_csv/00001_S2.csv').iloc[:, :len(checkbox_state)]
        show_df = select_show_columns(show_df, checkbox_state)  if True in checkbox_state else pd.DataFrame([['Select'],['one'],['column']]).T
        self.model = pandasModel(show_df)
        self.nw.tableView.setModel(self.model)
        header = self.nw.tableView.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def update_figure(self):
        fig_list = [self.nw.figure, self.nw.figure_2]
        self.figA, self.figB = self.analyze_history[self.checklist.index(True)][2:4]
        view_width, view_height = self.nw.figure.width(), self.nw.figure.height()
        for i, fig in enumerate([self.figA, self.figB]):
            fig.set_size_inches(view_width / 110, view_height / 120)
            img = plt_figure2array(fig)
            qt_img = convert_cv_qt(img, view_height)
            round_qt_img = cut2round(qt_img)
            fig_list[i].setPixmap(round_qt_img)
            