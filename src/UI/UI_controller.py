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

from src.UI.UI_library import pandasModel, select_show_columns
from src.UI.UI import Ui_MainWindow
from src.UI.csv_page import Ui_new_page
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    def __init__(self, source):
        super().__init__()
        self.source = source
        # QThread.
    def run(self):
        cap = cv2.VideoCapture(self.source)
        while(True):
            ret, frame = cap.read()
            if type(self.source) != int:
                cv2.waitKey(20)
            if ret:
                self.change_pixmap_signal.emit(frame)

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("badminton demo")
        self.setup_control()

    def setup_control(self):
        self.ui.select_file.clicked.connect(self.open_video)
        self.ui.show_csv.clicked.connect(self.open_page)

    def setup_nw_control(self):
        self.checkboxes = [self.nw.checkBox, self.nw.checkBox_2, self.nw.checkBox_3, self.nw.checkBox_4, self.nw.checkBox_5, self.nw.checkBox_6]
        for box in self.checkboxes:
            box.clicked.connect(self.change_csv)
           
    def open_video(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Open file", "./", filter = "*.avi *.mp4 *.wmv *.mov")# start path 
        if len(filename) == 0:
            self.ui.show_path.setText("Select a target video...")
        else:
            self.ui.show_path.setText(filename)
    
    def open_page(self):
        self.newWindow = QtWidgets.QMainWindow()
        self.nw = Ui_new_page()
        self.nw.setupUi(self.newWindow)
        self.newWindow.show()
        self.setup_nw_control()

    def change_csv(self):
        check_state = [box.isChecked() for box in self.checkboxes]
        df = pd.read_csv('saved_csv/00001_S2.csv').iloc[:, :len(check_state)]
        show_df = select_show_columns(df, check_state)  if True in check_state else pd.DataFrame([['Select'],['one'],['column']]).T
        self.model = pandasModel(show_df)
        self.nw.tableView.setModel(self.model)
        header = self.nw.tableView.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

