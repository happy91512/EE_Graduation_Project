# Form implementation generated from reading ui file 'csv_page.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_new_page(object):
    def setupUi(self, new_page):
        new_page.setObjectName("new_page")
        new_page.resize(1080, 720)
        self.verticalFrame = QtWidgets.QFrame(parent=new_page)
        self.verticalFrame.setGeometry(QtCore.QRect(20, 20, 1041, 80))
        self.verticalFrame.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.verticalFrame.setObjectName("verticalFrame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalFrame)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayoutWidget = QtWidgets.QWidget(parent=new_page)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 110, 101, 591))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.checkBox = QtWidgets.QCheckBox(parent=self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.checkBox.setFont(font)
        self.checkBox.setObjectName("checkBox")
        self.verticalLayout_2.addWidget(self.checkBox)
        self.checkBox_2 = QtWidgets.QCheckBox(parent=self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.checkBox_2.setFont(font)
        self.checkBox_2.setObjectName("checkBox_2")
        self.verticalLayout_2.addWidget(self.checkBox_2)
        self.checkBox_3 = QtWidgets.QCheckBox(parent=self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.checkBox_3.setFont(font)
        self.checkBox_3.setObjectName("checkBox_3")
        self.verticalLayout_2.addWidget(self.checkBox_3)
        self.checkBox_4 = QtWidgets.QCheckBox(parent=self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.checkBox_4.setFont(font)
        self.checkBox_4.setObjectName("checkBox_4")
        self.verticalLayout_2.addWidget(self.checkBox_4)
        self.checkBox_5 = QtWidgets.QCheckBox(parent=self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.checkBox_5.setFont(font)
        self.checkBox_5.setObjectName("checkBox_5")
        self.verticalLayout_2.addWidget(self.checkBox_5)
        self.checkBox_6 = QtWidgets.QCheckBox(parent=self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.checkBox_6.setFont(font)
        self.checkBox_6.setObjectName("checkBox_6")
        self.verticalLayout_2.addWidget(self.checkBox_6)
        self.tableView = QtWidgets.QTableView(parent=new_page)
        self.tableView.setGeometry(QtCore.QRect(130, 110, 931, 591))
        self.tableView.setStyleSheet("background:rgb(240, 240, 240)")
        self.tableView.setObjectName("tableView")

        self.retranslateUi(new_page)
        QtCore.QMetaObject.connectSlotsByName(new_page)

    def retranslateUi(self, new_page):
        _translate = QtCore.QCoreApplication.translate
        new_page.setWindowTitle(_translate("new_page", "Form"))
        self.checkBox.setText(_translate("new_page", "column1"))
        self.checkBox_2.setText(_translate("new_page", "column2"))
        self.checkBox_3.setText(_translate("new_page", "column3"))
        self.checkBox_4.setText(_translate("new_page", "column4"))
        self.checkBox_5.setText(_translate("new_page", "column5"))
        self.checkBox_6.setText(_translate("new_page", "column6"))
