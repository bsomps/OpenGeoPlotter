# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ColumnSelectionDialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ColumnSelectionDialog(object):
    def setupUi(self, ColumnSelectionDialog):
        ColumnSelectionDialog.setObjectName("ColumnSelectionDialog")
        ColumnSelectionDialog.resize(568, 531)
        ColumnSelectionDialog.setModal(True)
        self.column_listWidget = QtWidgets.QListWidget(ColumnSelectionDialog)
        self.column_listWidget.setGeometry(QtCore.QRect(120, 160, 256, 192))
        self.column_listWidget.setObjectName("column_listWidget")
        self.plot_button = QtWidgets.QPushButton(ColumnSelectionDialog)
        self.plot_button.setGeometry(QtCore.QRect(140, 80, 84, 25))
        self.plot_button.setCheckable(True)
        self.plot_button.setObjectName("plot_button")

        self.retranslateUi(ColumnSelectionDialog)
        QtCore.QMetaObject.connectSlotsByName(ColumnSelectionDialog)

    def retranslateUi(self, ColumnSelectionDialog):
        _translate = QtCore.QCoreApplication.translate
        ColumnSelectionDialog.setWindowTitle(_translate("ColumnSelectionDialog", "Dialog"))
        self.plot_button.setText(_translate("ColumnSelectionDialog", "Plot"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ColumnSelectionDialog = QtWidgets.QDialog()
    ui = Ui_ColumnSelectionDialog()
    ui.setupUi(ColumnSelectionDialog)
    ColumnSelectionDialog.show()
    sys.exit(app.exec_())
