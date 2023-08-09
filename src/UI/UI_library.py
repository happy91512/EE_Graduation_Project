import pandas as pd
import cv2
import sys
import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtCore import QAbstractTableModel, Qt
from PyQt6.QtGui import QImage, QPixmap, QPainter
# from src.UI.UI_controller import VideoThread
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
matplotlib.use("Qt5Agg")

sys.path.append(str(Path(__file__).resolve().parents[2]))
class pandasModel(QAbstractTableModel):
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid():
            if role == Qt.ItemDataRole.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._data.columns[col]
        return None

def select_show_columns(df : pd.DataFrame, check_list : list[bool]) -> pd.DataFrame:
    show = df.iloc[:, check_list]
    return show

def convert_cv_qt(cv_img : np.ndarray, target_h):
    """Convert from an opencv image to QPixmap"""
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = cv_img.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    p = convert_to_Qt_format.scaledToHeight(target_h)
    return QPixmap.fromImage(p)

def cut2round(qt_img: QPixmap):
    round_qt_img = QPixmap(qt_img.size())
    round_qt_img.fill(Qt.GlobalColor.transparent)
    painter = QPainter(round_qt_img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(Qt.GlobalColor.black)
    painter.drawRoundedRect(qt_img.rect(), 10, 10)  
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.drawPixmap(qt_img.rect(), qt_img, qt_img.rect())
    painter.end()
    return round_qt_img

def check_video_thread(thread):
    if thread == None: 
        return
    elif thread.isRunning():
        thread.stop = True
    
def get_time_format(frame_count, fps):
    secs = int(frame_count/fps)
    mins = int(secs / 60)
    secs = secs % 60
    return f"{mins:02d}:{secs:02d}"

# def show_fig(graphicsView : QtWidgets.QGraphicsView, canvas : FigureCanvas):
#     graphicscene = QtWidgets.QGraphicsScene()
#     graphicscene.
#     graphicsView.setScene(graphicscene)

def plt_figure2array(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return image_array

if __name__ == '__main__':
    csv_path = 'saved_csv/00001_S2.csv'
    check_box = [True, False, False, True, True]
    df = pd.read_csv(csv_path).iloc[:, :5]
    show = df.iloc[:, check_box]
    print(show)
