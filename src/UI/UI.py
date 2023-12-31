# Form implementation generated from reading ui file 'UI.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1080, 720)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setDocumentMode(False)
        MainWindow.setTabShape(QtWidgets.QTabWidget.TabShape.Rounded)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.show_csv = QtWidgets.QPushButton(parent=self.centralwidget)
        self.show_csv.setGeometry(QtCore.QRect(950, 20, 111, 51))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("docs.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.show_csv.setIcon(icon)
        self.show_csv.setIconSize(QtCore.QSize(32, 32))
        self.show_csv.setObjectName("show_csv")
        self.select_file = QtWidgets.QPushButton(parent=self.centralwidget)
        self.select_file.setGeometry(QtCore.QRect(820, 20, 111, 51))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("folder.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.select_file.setIcon(icon1)
        self.select_file.setIconSize(QtCore.QSize(32, 32))
        self.select_file.setObjectName("select_file")
        self.show_path = QtWidgets.QTextBrowser(parent=self.centralwidget)
        self.show_path.setGeometry(QtCore.QRect(20, 20, 781, 51))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.show_path.setFont(font)
        self.show_path.setAutoFillBackground(False)
        self.show_path.setStyleSheet("background:rgb(220, 220, 220)")
        self.show_path.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.show_path.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        self.show_path.setAutoFormatting(QtWidgets.QTextEdit.AutoFormattingFlag.AutoBulletList)
        self.show_path.setObjectName("show_path")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.show_csv.setText(_translate("MainWindow", "預測成果"))
        self.select_file.setText(_translate("MainWindow", "選擇檔案"))
        self.show_path.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:\'.AppleSystemUIFont\'; font-size:18pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Select a target video...</p></body></html>"))
