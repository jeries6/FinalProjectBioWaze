# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:45:16 2021

@author: Jeries
"""

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
import concurrent
import os

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QFileDialog, QDialog, QWidget, QMessageBox
from model_controller import DataModel
from PyQt5.QtCore import QThread, pyqtSignal, QObject
import time


    
    
class TrainingThread(QThread):
    Result = None
    change_value = pyqtSignal(int)
    finished = pyqtSignal()

    def run(self):
        num = 100
        split_size = 0.25
        blocks = 128
        filters = 64
        pool_size = 1
        kernel = 5
        dropout = 0.25
        epochs = 4
        batch_size = 30
        controller = DataModel()
        for i in range(10):
            self.change_value.emit(i)
            time.sleep(0.05)
        mDataSet = controller.createDataSet(num)
        counter = 10
        while (counter < 25):
            self.change_value.emit(counter)
            time.sleep(0.04)
            counter += 1
        mDataSet = controller.normalize_data(mDataSet)
        while (counter < 32):
            self.change_value.emit(counter)
            time.sleep(0.04)
            counter += 1
        X, Y = controller.reshape_data(mDataSet)
        while (counter < 55):
            self.change_value.emit(counter)
            time.sleep(0.05)
            counter += 1
        X_train, X_test, Y_train, Y_test = controller.split_data(X, Y, split_size)
        model = controller.create_cnnLstm_model(blocks, filters, pool_size, kernel, dropout)
        while (counter < 70):
            self.change_value.emit(counter)
            time.sleep(0.04)
            counter += 1
        history = controller.train_cnn_lstm(model, X_train, Y_train, epochs, batch_size)
        while (counter < 101):
            self.change_value.emit(counter)
            time.sleep(0.05)
            counter += 1
        # self.Result[0] = history
        gHistory[0] = history
        self.finished.emit()


class Ui_MainWindow(QWidget):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1020, 796)
        MainWindow.setWindowIcon(QIcon(r'C:\Users\Jeries\Desktop\bioWaze\FinalProjectBioWaze\mIcon.png'))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.cnnLstmWidget = QtWidgets.QWidget(self.centralwidget)
        self.cnnLstmWidget.setEnabled(True)
        self.cnnLstmWidget.setGeometry(QtCore.QRect(10, 270, 801, 80))
        self.cnnLstmWidget.setObjectName("cnnLstmWidget")
        self.layoutWidget = QtWidgets.QWidget(self.cnnLstmWidget)
        self.layoutWidget.setGeometry(QtCore.QRect(20, 20, 751, 54))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_6 = QtWidgets.QLabel(self.layoutWidget)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_5.addWidget(self.label_6)
        self.cnnKernelEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.cnnKernelEdit.setObjectName("cnnKernelEdit")
        self.verticalLayout_5.addWidget(self.cnnKernelEdit)
        self.horizontalLayout_4.addLayout(self.verticalLayout_5)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_7 = QtWidgets.QLabel(self.layoutWidget)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_6.addWidget(self.label_7)
        self.cnnFiltersEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.cnnFiltersEdit.setObjectName("cnnFiltersEdit")
        self.verticalLayout_6.addWidget(self.cnnFiltersEdit)
        self.horizontalLayout_4.addLayout(self.verticalLayout_6)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_8 = QtWidgets.QLabel(self.layoutWidget)
        self.label_8.setObjectName("label_8")
        self.verticalLayout_7.addWidget(self.label_8)
        self.poolingSizeEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.poolingSizeEdit.setObjectName("poolingSizeEdit")
        self.verticalLayout_7.addWidget(self.poolingSizeEdit)
        self.horizontalLayout_4.addLayout(self.verticalLayout_7)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.label_9 = QtWidgets.QLabel(self.layoutWidget)
        self.label_9.setObjectName("label_9")
        self.verticalLayout_8.addWidget(self.label_9)
        self.lstmBlocksEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.lstmBlocksEdit.setObjectName("lstmBlocksEdit")
        self.verticalLayout_8.addWidget(self.lstmBlocksEdit)
        self.horizontalLayout_4.addLayout(self.verticalLayout_8)
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.label_10 = QtWidgets.QLabel(self.layoutWidget)
        self.label_10.setObjectName("label_10")
        self.verticalLayout_9.addWidget(self.label_10)
        self.dropoutEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.dropoutEdit.setObjectName("dropoutEdit")
        self.verticalLayout_9.addWidget(self.dropoutEdit)
        self.horizontalLayout_4.addLayout(self.verticalLayout_9)
        self.widget_2 = QtWidgets.QWidget(self.centralwidget)
        self.widget_2.setGeometry(QtCore.QRect(30, 370, 261, 80))
        self.widget_2.setObjectName("widget_2")
        self.layoutWidget1 = QtWidgets.QWidget(self.widget_2)
        self.layoutWidget1.setGeometry(QtCore.QRect(30, 20, 216, 27))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_11 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_6.addWidget(self.label_11)
        self.nNeighborsEdit = QtWidgets.QLineEdit(self.layoutWidget1)
        self.nNeighborsEdit.setObjectName("nNeighborsEdit")
        self.horizontalLayout_6.addWidget(self.nNeighborsEdit)
        self.svmWidget = QtWidgets.QWidget(self.centralwidget)
        self.svmWidget.setGeometry(QtCore.QRect(350, 370, 501, 80))
        self.svmWidget.setObjectName("svmWidget")
        self.layoutWidget2 = QtWidgets.QWidget(self.svmWidget)
        self.layoutWidget2.setGeometry(QtCore.QRect(40, 20, 355, 54))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.layoutWidget2)
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout()
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.label_12 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_12.setObjectName("label_12")
        self.verticalLayout_13.addWidget(self.label_12)
        self.svmKernelCombo = QtWidgets.QComboBox(self.layoutWidget2)
        self.svmKernelCombo.setObjectName("svmKernelCombo")
        self.svmKernelCombo.addItem("")
        self.svmKernelCombo.addItem("")
        self.svmKernelCombo.addItem("")
        self.svmKernelCombo.addItem("")
        self.svmKernelCombo.addItem("")
        self.verticalLayout_13.addWidget(self.svmKernelCombo)
        self.horizontalLayout_7.addLayout(self.verticalLayout_13)
        self.verticalLayout_14 = QtWidgets.QVBoxLayout()
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.label_13 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_13.setObjectName("label_13")
        self.verticalLayout_14.addWidget(self.label_13)
        self.gammaCombo = QtWidgets.QComboBox(self.layoutWidget2)
        self.gammaCombo.setObjectName("gammaCombo")
        self.gammaCombo.addItem("")
        self.gammaCombo.addItem("")
        self.verticalLayout_14.addWidget(self.gammaCombo)
        self.horizontalLayout_7.addLayout(self.verticalLayout_14)
        self.verticalLayout_15 = QtWidgets.QVBoxLayout()
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.label_14 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_14.setObjectName("label_14")
        self.verticalLayout_15.addWidget(self.label_14)
        self.svmCEdit = QtWidgets.QLineEdit(self.layoutWidget2)
        self.svmCEdit.setObjectName("svmCEdit")
        self.verticalLayout_15.addWidget(self.svmCEdit)
        self.horizontalLayout_7.addLayout(self.verticalLayout_15)
        self.trainModelButton = QtWidgets.QPushButton(self.centralwidget)
        self.trainModelButton.setGeometry(QtCore.QRect(340, 490, 171, 51))
        self.trainModelButton.setObjectName("trainModelButton")
        self.trainingProgressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.trainingProgressBar.setGeometry(QtCore.QRect(130, 570, 701, 23))
        self.trainingProgressBar.setProperty("value", 0)
        self.trainingProgressBar.setObjectName("trainingProgressBar")
        self.saveModelButton = QtWidgets.QPushButton(self.centralwidget)
        self.saveModelButton.setGeometry(QtCore.QRect(0, 730, 1021, 41))
        self.saveModelButton.setObjectName("saveModelButton")
        self.layoutWidget3 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget3.setGeometry(QtCore.QRect(20, 630, 197, 73))
        self.layoutWidget3.setObjectName("layoutWidget3")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.layoutWidget3)
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_15 = QtWidgets.QLabel(self.layoutWidget3)
        self.label_15.setObjectName("label_15")
        self.horizontalLayout_8.addWidget(self.label_15)
        self.accuracyEdit = QtWidgets.QPlainTextEdit(self.layoutWidget3)
        self.accuracyEdit.setReadOnly(True)
        self.accuracyEdit.setObjectName("accuracyEdit")
        self.horizontalLayout_8.addWidget(self.accuracyEdit)
        self.layoutWidget4 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget4.setGeometry(QtCore.QRect(30, 140, 89, 52))
        self.layoutWidget4.setObjectName("layoutWidget4")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.layoutWidget4)
        self.verticalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.label_5 = QtWidgets.QLabel(self.layoutWidget4)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_10.addWidget(self.label_5)
        self.modelComboBox = QtWidgets.QComboBox(self.layoutWidget4)
        self.modelComboBox.setObjectName("modelComboBox")
        self.modelComboBox.addItem("")
        self.modelComboBox.addItem("")
        self.modelComboBox.addItem("")
        self.verticalLayout_10.addWidget(self.modelComboBox)
        self.layoutWidget5 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget5.setGeometry(QtCore.QRect(20, 20, 961, 91))
        self.layoutWidget5.setObjectName("layoutWidget5")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget5)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout()
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.label = QtWidgets.QLabel(self.layoutWidget5)
        self.label.setObjectName("label")
        self.verticalLayout_12.addWidget(self.label)
        self.browseButton = QtWidgets.QPushButton(self.layoutWidget5)
        self.browseButton.setObjectName("browseButton")
        self.verticalLayout_12.addWidget(self.browseButton)
        self.dataLabel1 = QtWidgets.QLabel(self.layoutWidget5)
        self.dataLabel1.setObjectName("dataLabel1")
        self.verticalLayout_12.addWidget(self.dataLabel1)
        self.horizontalLayout.addLayout(self.verticalLayout_12)
        spacerItem = QtWidgets.QSpacerItem(100, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.splitLabel = QtWidgets.QLabel(self.layoutWidget5)
        self.splitLabel.setObjectName("splitLabel")
        self.horizontalLayout_5.addWidget(self.splitLabel)
        self.label_4 = QtWidgets.QLabel(self.layoutWidget5)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_5.addWidget(self.label_4)
        self.verticalLayout_11.addLayout(self.horizontalLayout_5)
        self.splitSlider = QtWidgets.QSlider(self.layoutWidget5)
        self.splitSlider.setProperty("value", 80)
        self.splitSlider.setOrientation(QtCore.Qt.Horizontal)
        self.splitSlider.setObjectName("splitSlider")
        self.verticalLayout_11.addWidget(self.splitSlider)
        self.horizontalLayout.addLayout(self.verticalLayout_11)
        spacerItem1 = QtWidgets.QSpacerItem(250, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.batchLabel = QtWidgets.QLabel(self.layoutWidget5)
        self.batchLabel.setObjectName("batchLabel")
        self.horizontalLayout.addWidget(self.batchLabel)
        self.batchEdit = QtWidgets.QLineEdit(self.layoutWidget5)
        self.batchEdit.setObjectName("batchEdit")
        self.horizontalLayout.addWidget(self.batchEdit)
        self.epochsLabel = QtWidgets.QLabel(self.layoutWidget5)
        self.epochsLabel.setObjectName("epochsLabel")
        self.horizontalLayout.addWidget(self.epochsLabel)
        self.epochsEdit = QtWidgets.QLineEdit(self.layoutWidget5)
        self.epochsEdit.setObjectName("epochsEdit")
        self.horizontalLayout.addWidget(self.epochsEdit)
        self.layoutWidget6 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget6.setGeometry(QtCore.QRect(270, 140, 521, 81))
        self.layoutWidget6.setObjectName("layoutWidget6")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget6)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.groupBox = QtWidgets.QGroupBox(self.layoutWidget6)
        self.groupBox.setObjectName("groupBox")
        self.radioButton_3 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_3.setGeometry(QtCore.QRect(10, 20, 111, 20))
        self.radioButton_3.setObjectName("radioButton_3")
        self.radioButton_4 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_4.setGeometry(QtCore.QRect(10, 40, 91, 20))
        self.radioButton_4.setObjectName("radioButton_4")
        self.horizontalLayout_2.addWidget(self.groupBox)
        self.usersNumWidg = QtWidgets.QWidget(self.layoutWidget6)
        self.usersNumWidg.setMaximumSize(QtCore.QSize(16777215, 2000))
        self.usersNumWidg.setObjectName("usersNumWidg")
        self.layoutWidget7 = QtWidgets.QWidget(self.usersNumWidg)
        self.layoutWidget7.setGeometry(QtCore.QRect(30, 20, 135, 46))
        self.layoutWidget7.setObjectName("layoutWidget7")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget7)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.numLabel = QtWidgets.QLabel(self.layoutWidget7)
        self.numLabel.setObjectName("numLabel")
        self.verticalLayout.addWidget(self.numLabel)
        self.usersNumEdit = QtWidgets.QLineEdit(self.layoutWidget7)
        self.usersNumEdit.setObjectName("usersNumEdit")
        self.verticalLayout.addWidget(self.usersNumEdit)
        self.horizontalLayout_2.addWidget(self.usersNumWidg)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.modelComboBox.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.controller = DataModel()
        
        self.saveModelButton.clicked.connect(self.saveModel)
        self.browseButton.clicked.connect(self.browseButtonClick)
        self.splitSlider.valueChanged.connect(self.splitSliderChanged)
        self.modelComboBox.currentIndexChanged.connect(self.modelSelectionChanged)
        self.trainModelButton.clicked.connect(self.train_model_button)
        self.widget_2.setVisible(False)
        self.svmWidget.setVisible(False)
        self.usersNumWidg.setVisible(False)
        self.radioButton_3.toggled.connect(self.dataSimRadio)
        self.radioButton_4.toggled.connect(self.dataUpRadio)
        
        #set default values
        
        self.cnnKernelEdit.setText('5')
        self.cnnFiltersEdit.setText('4')
        self.poolingSizeEdit.setText('1')
        self.lstmBlocksEdit.setText('5')
        self.dropoutEdit.setText('0.2')
        self.batchEdit.setText('30')



    def saveModel(self):
        self.controller.save_model(self.model)
        import os
        cwd = os.getcwd()
        msg = QMessageBox()
        msg.setWindowTitle("Success!")
        msg.setText(f"The Trained Model has been successfully saved to: {cwd}")
        msg.setIcon(QMessageBox.Information)
        
        x = msg.exec_()

    def dataSimRadio(self):
        self.usersNumWidg.setVisible(True)
    
    def dataUpRadio(self):
        self.usersNumWidg.setVisible(False)

    def browseButtonClick(self):
        fname = QFileDialog.getOpenFileName(self, 'open file', '.')
        head, tail = os.path.split(fname[0])
        self.dataLabel1.setText(tail)
        self.fileName = fname[0]

    def splitSliderChanged(self):
        self.label_4.setText(str(self.splitSlider.value()) + '%')

    def modelSelectionChanged(self):
        if (self.modelComboBox.currentText() == 'KNN'):
            self.cnnLstmWidget.setVisible(False)
            self.svmWidget.setVisible(False)
            self.widget_2.setVisible(True)
        if (self.modelComboBox.currentText() == 'SVM'):
            self.cnnLstmWidget.setVisible(False)
            self.svmWidget.setVisible(True)
            self.widget_2.setVisible(False)
        if (self.modelComboBox.currentText() == 'CNN-LSTM'):
            self.cnnLstmWidget.setVisible(True)
            self.svmWidget.setVisible(False)
            self.widget_2.setVisible(False)

    def train_model_button(self):
        result = [None]
        
        if(str(self.modelComboBox.currentText()) == "CNN-LSTM"):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                f1 = executor.submit(self.workerThread, int(self.usersNumEdit.text()), self.splitSlider.value(), int(self.lstmBlocksEdit.text()), int(self.cnnFiltersEdit.text()), int(self.poolingSizeEdit.text()), int(self.cnnKernelEdit.text()), float(self.dropoutEdit.text()), int(self.epochsEdit.text()), int(self.batchEdit.text()))
                mHistory = f1.result()
                nl = '\n'
                acc = mHistory.history['accuracy'][-1]
                acc = "{:.5f}".format(acc)
                loss = mHistory.history['loss'][-1]
                loss = "{:.5f}".format(loss)
                plainText = f"Accuracy:{nl} {acc} {nl} Loss:{nl} {loss}"
                self.accuracyEdit.appendPlainText(str(plainText))
                
        
            
        elif(str(self.modelComboBox.currentText()) == "KNN"):    
            counter = 0
            while (counter < 37):
                self.setProgressBar(counter)
                time.sleep(0.04)
                counter += 1
            KNNacc = self.controller.knn_model(self.fileName, int(self.nNeighborsEdit.text()))
            while (counter < 101):
                self.setProgressBar(counter)
                time.sleep(0.04)
                counter += 1
            KNNacc = "{:.5f}".format(KNNacc)
            nl = '\n'
            plainText = f"KNN-Accuracy:{nl} {KNNacc}"
            self.accuracyEdit.appendPlainText(str(plainText))

        elif(self.modelComboBox.currentText() == "SVM"): 
            counter = 0
            while (counter < 43):
                self.setProgressBar(counter)
                time.sleep(0.04)
                counter += 1
            print(self.fileName, str(self.svmKernelCombo.currentText()), str(self.gammaCombo.currentText()), int(self.svmCEdit.text()))
            SVMacc = self.controller.svm_model(self.fileName, str(self.svmKernelCombo.currentText()), str(self.gammaCombo.currentText()), int(self.svmCEdit.text()))
            while (counter < 101):
                self.setProgressBar(counter)
                time.sleep(0.04)
                counter += 1
            SVMacc = "{:.5f}".format(SVMacc)
            nl = '\n'
            plainText = f"SVM-Accuracy:{nl} {SVMacc}"
            self.accuracyEdit.appendPlainText(str(plainText))

       
        
        
    def workerThread(self, num, split_size, blocks, filters, pool_size, kernel, dropout, epochs, batch_size):
        print(num,split_size, blocks, filters, pool_size, kernel, dropout, epochs, batch_size)
        #controller = DataModel()
        for i in range(10):
            self.setProgressBar(i)
            time.sleep(0.05)
        mDataSet = self.controller.createDataSet(num)
        newData = self.controller.convert_avg(mDataSet)
        mDataSet.to_csv (r'C:\Users\Jeries\Desktop\bioWaze\FinalProjectBioWaze\SavedDataSets\DataSet.csv', index = False, header=True)
        newData.to_csv (r'C:\Users\Jeries\Desktop\bioWaze\FinalProjectBioWaze\SavedDataSets\DataSet_Avg.csv', index = False, header=True)
                
  
        counter = 10
        while (counter < 25):
            self.setProgressBar(counter)
            time.sleep(0.04)
            counter += 1
        mDataSet = self.controller.normalize_data(mDataSet)
        while (counter < 32):
            self.setProgressBar(counter)
            time.sleep(0.04)
            counter += 1
        X, Y = self.controller.reshape_data(mDataSet)
        while (counter < 55):
            self.setProgressBar(counter)
            time.sleep(0.05)
            counter += 1
        X_train, X_test, Y_train, Y_test = self.controller.split_data(X, Y, split_size)
        self.model = self.controller.create_cnnLstm_model(blocks, filters, pool_size, kernel, dropout)
        while (counter < 70):
            self.setProgressBar(counter)
            time.sleep(0.04)
            counter += 1
        history = self.controller.train_cnn_lstm(self.model, X_train, Y_train, epochs, batch_size)
        while (counter < 101):
            self.setProgressBar(counter)
            time.sleep(0.05)
            counter += 1
        return history

    def setProgressBar(self, val):
        self.trainingProgressBar.setValue(val)

    def getModelHistory(self):
        pass

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Data Scientist"))
        self.label_6.setText(_translate("MainWindow", "CNN Kernel:"))
        self.label_7.setText(_translate("MainWindow", "CNN Filters:"))
        self.label_8.setText(_translate("MainWindow", "Pooling Size:"))
        self.label_9.setText(_translate("MainWindow", "LSTM Blocks:"))
        self.label_10.setText(_translate("MainWindow", "Dropout:"))
        self.label_11.setText(_translate("MainWindow", "n_neighbors:"))
        self.label_12.setText(_translate("MainWindow", "Kernel:"))
        self.svmKernelCombo.setItemText(0, _translate("MainWindow", "linear"))
        self.svmKernelCombo.setItemText(1, _translate("MainWindow", "poly"))
        self.svmKernelCombo.setItemText(2, _translate("MainWindow", "rbf"))
        self.svmKernelCombo.setItemText(3, _translate("MainWindow", "sigmoid"))
        self.svmKernelCombo.setItemText(4, _translate("MainWindow", "precomputed"))
        self.label_13.setText(_translate("MainWindow", "Gamma:"))
        self.gammaCombo.setItemText(0, _translate("MainWindow", "auto"))
        self.gammaCombo.setItemText(1, _translate("MainWindow", "scale"))
        self.label_14.setText(_translate("MainWindow", "C:"))
        self.trainModelButton.setText(_translate("MainWindow", "Train Model"))
        self.saveModelButton.setText(_translate("MainWindow", "Save Model"))
        self.label_15.setText(_translate("MainWindow", "Accuracy:"))
        self.label_5.setText(_translate("MainWindow", "Model:"))
        self.modelComboBox.setCurrentText(_translate("MainWindow", "CNN-LSTM"))
        self.modelComboBox.setItemText(0, _translate("MainWindow", "CNN-LSTM"))
        self.modelComboBox.setItemText(1, _translate("MainWindow", "KNN"))
        self.modelComboBox.setItemText(2, _translate("MainWindow", "SVM"))
        self.label.setText(_translate("MainWindow", "Select Data(CSV File):"))
        self.browseButton.setText(_translate("MainWindow", "Browse"))
        self.dataLabel1.setText(_translate("MainWindow", "No Data Yet."))
        self.splitLabel.setText(_translate("MainWindow", "Train Data Split:"))
        self.label_4.setText(_translate("MainWindow", "80%"))
        self.batchLabel.setText(_translate("MainWindow", "Batch Size:"))
        self.epochsLabel.setText(_translate("MainWindow", "#Epochs:"))
        self.groupBox.setTitle(_translate("MainWindow", "Simulate/Upload Data"))
        self.radioButton_3.setText(_translate("MainWindow", "Simulate Data"))
        self.radioButton_4.setText(_translate("MainWindow", "Upload Data"))
        self.numLabel.setText(_translate("MainWindow", "Number of users:"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

