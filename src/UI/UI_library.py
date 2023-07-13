import pandas as pd
import cv2
import numpy as np
from PyQt6.QtCore import QAbstractTableModel, Qt
from PyQt6.QtGui import QImage, QPixmap

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

if __name__ == '__main__':
    csv_path = 'saved_csv/00001_S2.csv'
    check_box = [True, False, False, True, True]
    df = pd.read_csv(csv_path).iloc[:, :5]
    show = df.iloc[:, check_box]
    print(show)
    # select_show_columns