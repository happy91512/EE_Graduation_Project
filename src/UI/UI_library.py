import pandas as pd
import sys
from PyQt6.QtWidgets import QApplication, QTableView
from PyQt6.QtCore import QAbstractTableModel, Qt


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


if __name__ == '__main__':
    csv_path = 'saved_csv/00001_S2.csv'
    check_box = [True, False, False, True, True]
    df = pd.read_csv(csv_path).iloc[:, :5]
    show = df.iloc[:, check_box]
    print(show)
    # select_show_columns