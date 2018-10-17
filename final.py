import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from model import FERModel
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
from PyQt5.QtGui import QIcon

# Initialize the OpenCV VideoCapture
rgb = cv2.VideoCapture(0)

#faceCascade contains facial Cascade Classifier loaded from xml file
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize global close_cam variable to 0,
# When close_cam == 1, camera closes
close_cam = 0

# ------------------------USER INTERFACE START----------------------------
# Main User Interface of Application develped using PyQT5
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(650, 493)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(110, 0, 431, 61))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(60, 60, 481, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.camera_btn = QtWidgets.QPushButton(self.centralwidget)
        self.camera_btn.setGeometry(QtCore.QRect(150, 160, 111, 31))
        self.camera_btn.setObjectName("camera_btn")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(290, 160, 231, 21))
        self.label_2.setObjectName("label_2")
        self.camera_btn2 = QtWidgets.QPushButton(self.centralwidget)
        self.camera_btn2.setGeometry(QtCore.QRect(150, 210, 111, 31))
        self.camera_btn2.setObjectName("camera_btn2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(290, 210, 231, 21))
        self.label_3.setObjectName("label_3")
        self.image_btn1 = QtWidgets.QPushButton(self.centralwidget)
        self.image_btn1.setGeometry(QtCore.QRect(150, 260, 111, 31))
        self.image_btn1.setObjectName("image_btn1")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(290, 260, 231, 21))
        self.label_4.setObjectName("label_4")
        self.image_btn2 = QtWidgets.QPushButton(self.centralwidget)
        self.image_btn2.setGeometry(QtCore.QRect(150, 310, 111, 31))
        self.image_btn2.setObjectName("image_btn2")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(290, 310, 231, 21))
        self.label_5.setObjectName("label_5")
        self.image_btn3 = QtWidgets.QPushButton(self.centralwidget)
        self.image_btn3.setGeometry(QtCore.QRect(150, 360, 111, 31))
        self.image_btn3.setObjectName("image_btn3")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(290, 360, 231, 21))
        self.label_6.setObjectName("label_6")
        self.close_btn = QtWidgets.QPushButton(self.centralwidget)
        self.close_btn.setGeometry(QtCore.QRect(150, 410, 111, 31))
        self.close_btn.setObjectName("close_btn")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(290, 410, 231, 21))
        self.label_7.setObjectName("label_7")
        self.radioButton = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton.setGeometry(QtCore.QRect(290, 100, 91, 17))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_2.setGeometry(QtCore.QRect(390, 100, 82, 17))
        self.radioButton_2.setObjectName("radioButton_2")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(110, 100, 161, 21))
        self.label_8.setObjectName("label_8")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 650, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.camera_btn.clicked.connect(self.start)
        self.camera_btn2.clicked.connect(self.stop_cam_func)
        self.image_btn1.clicked.connect(self.open_image)
        self.image_btn2.clicked.connect(self.show_img)
        self.image_btn3.clicked.connect(self.start_img)
        self.close_btn.clicked.connect(QtCore.QCoreApplication.instance().quit)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Mood detection"))
        self.label.setText(_translate("MainWindow", "Mood Detection using Neural Network"))
        self.camera_btn.setText(_translate("MainWindow", "Open Camera"))
        self.label_2.setText(_translate("MainWindow", "Open Camera to detect emotion in real time"))
        self.camera_btn2.setText(_translate("MainWindow", "Close Camera"))
        self.label_3.setText(_translate("MainWindow", "Close the Camera feed"))
        self.image_btn1.setText(_translate("MainWindow", "Load Image"))
        self.label_4.setText(_translate("MainWindow", "Select an image from file system"))
        self.image_btn2.setText(_translate("MainWindow", "Show Image"))
        self.label_5.setText(_translate("MainWindow", "Show/View the selected image"))
        self.image_btn3.setText(_translate("MainWindow", "Image prediction"))
        self.label_6.setText(_translate("MainWindow", "Predict mood of subjects in the loaded image"))
        self.close_btn.setText(_translate("MainWindow", "Close Application"))
        self.label_7.setText(_translate("MainWindow", "close this application"))
        self.radioButton.setText(_translate("MainWindow", "Shallow Model"))
        self.radioButton_2.setText(_translate("MainWindow", "Deep Model"))
        self.label_8.setText(_translate("MainWindow", "Select Model for Mood Detection"))

    #------------------USER INTERFACE END------------------------------------------


    # Action when "Load Image" button clicked
    # Load image and store in global name variable
    def open_image(self):
        global name
        name, _ = QtWidgets.QFileDialog.getOpenFileName()
        self.name=name

    # Action when "Show Image" button clicked
    def show_img(self):
        loadedImage = cv2.imread(self.name)
        plt.imshow(loadedImage, cmap='gray')
        plt.show()

    def convertToRGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Action when "Image Prediction" is clicked
    # Based on the radio button, model is loaded
    def start_img(self):
        if self.radioButton.isChecked==True:
            model = FERModel("shallow_model.json", "shallow_weights.h5")
        else:
            model = FERModel("deep_model.json", "deep_weights.h5")
        self.predict_img(model)


    # PREDICTION ON IMAGE
    def predict_img(self, cnn):

        # Loading image from disk
        loadedImage = cv2.imread(name)
        print(loadedImage.shape)

        # Converting image to grayscale
        gray_img = cv2.cvtColor(loadedImage, cv2.COLOR_BGR2GRAY)
        #plt.imshow(gray_img, cmap='gray')
        #plt.show()
        
        # Loading Haar Cascade from xml file
        haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # Detecting face(s) in the grayscale image
        faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
        print('Faces found: ', len(faces))

        # Creating FERModel object
        if self.radioButton.isChecked==True:
            model = FERModel("shallow_model.json", "shallow_weights.h5")
        else:
            model = FERModel("deep_model.json", "deep_weights.h5")

        # Looping through all the faces detected
        for (x, y, w, h) in faces:

            # originalFace contains region from x,y extended to height h and width w     
            originalFace = gray_img[y:y+h, x:x+w]
                        
            # roi is the Region Of Interest
            # Since our input matrix size is 48x48, for the model, We convert the face to this size
            roi = cv2.resize(originalFace, (48, 48))

            # predict_emotion returns the string emotion
            pred = cnn.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            print(pred)

            # Placing pred (returned emotion name) on the image
            cv2.putText(loadedImage, pred, (x, y), font, 1, (255, 255, 0), 2)

            # Enclosing the image in a box
            cv2.rectangle(loadedImage,(x,y),(x+w,y+h),(255,0,0),2)

        # Show the loaded image with emotion name and box
        plt.imshow(self.convertToRGB(loadedImage))
        plt.show()


    # Action when "Open Camera" button is clicked
    def start(self):
        print(close_cam)

        # create Facial Expression Model object
        if self.radioButton.isChecked==True:
            model = FERModel("shallow_model.json", "shallow_weights.h5")
        else:
            model = FERModel("deep_model.json", "deep_weights.h5")
        self.start_app(model)

    def __get_data__(self):
        _, fr = rgb.read()
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        return faces, fr, gray

    def start_app(self, cnn):
        skip_frame = 10
        data = []
        flag = False
        ix = 0

        # Keep camera window open till close_cam is 0
        while close_cam==0:
            print(close_cam)
            ix += 1
            
            faces, fr, gray_fr = self.__get_data__()
            for (x, y, w, h) in faces:
                fc = gray_fr[y:y+h, x:x+w]
                
                roi = cv2.resize(fc, (48, 48))
                pred = cnn.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

                cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

            if cv2.waitKey(1) == 27:
                break
            cv2.imshow('CNN Expression Prediction', fr)
        cv2.destroyAllWindows()
        self.change_close_cam_val()

    def stop_cam_func(self):
        global close_cam
        close_cam=1
        print('now close cam is {}'.format(close_cam))

    def change_close_cam_val(self):
        global close_cam
        close_cam=0



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())