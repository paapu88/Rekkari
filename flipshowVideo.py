import cv2
import numpy as np
import sys
from PyQt5 import QtWidgets, QtCore, QtGui


class QtCapture(QtWidgets.QWidget):
    def __init__(self, *args):
        super(QtWidgets.QWidget, self).__init__()

        self.fps = 24
        #self.cap = cv2.VideoCapture(*args)
        self.read_frames(sys.argv[1])
        self.frame_nr = 0
        #print ("cap", self.cap)
        self.video_frame = QtWidgets.QLabel()
        lay = QtWidgets.QVBoxLayout()
        #lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.video_frame)
        self.setLayout(lay)
        print ("INIT OK")

    def read_frames(self, filename):
        cap = cv2.VideoCapture(filename)
        self.rekkari_cascade = cv2.CascadeClassifier('./rekkari.xml')

        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frames = []
        print ("starting reading: filename", filename)
        while not cap.isOpened():
            cap = cv2.VideoCapture(filename)
            cv2.waitKey(1000)
            print("Wait for the header")


        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        while True:
            flag, frame = cap.read()
            if flag:
                # The frame is ready and already captured
                #cv2.imshow('video', frame)
                # Convert to RGB for QImage.
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
                frame = cv2.transpose(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rekkaris = self.rekkari_cascade.detectMultiScale(gray, 1.1, 5)
                for (x,y,w,h) in rekkaris:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
                self.frames.append(frame)
                pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                print (str(pos_frame)+" frames")
            else:
                # The next frame is not ready, so we try to read it again
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
                print("frame is not ready")
                # It is better to wait for a while for the next frame to be ready
                cv2.waitKey(1000)

            if cv2.waitKey(10) == 27:
                break
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                break
            if pos_frame > 15:
                break



        cap.release()
        print("finished reading")


    def setFPS(self, fps):
        self.fps = fps

    def nextFrameSlot(self):
        import sys

        print("frameinfo", self.frame_nr, len(self.frames))
        #sys.exit()
        if self.frame_nr < (len(self.frames) -1):
            self.frame_nr = self.frame_nr + 1
        else:
            self.frame_nr = 0
        print("frameinfo2", self.frame_nr, len(self.frames))
        frame = self.frames[self.frame_nr]

        # OpenCV yields frames in BGR format
        #frame = cv2.cvtColor(frame, cv2.cv.CV_BGR2RGB)
        #cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
        #img = frame["img"]
        if frame is not None:
            print('FRAME', frame)
            #height, width, bytesPerComponent = frame.shape
            #bytesPerLine = bytesPerComponent * width
            #img = QtGui.QImage(self.cv_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(img)
            self.video_frame.setPixmap(pix)


    def start(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(1000./self.fps)

    def stop(self):
        self.timer.stop()

    def deleteLater(self):
        self.cap.release()
        super(QtWidgets.QWidget, self).deleteLater()

class ControlWindow(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.capture = None

        self.start_button = QtWidgets.QPushButton('Start')
        self.start_button.clicked.connect(self.startCapture)
        self.quit_button = QtWidgets.QPushButton('End')
        self.quit_button.clicked.connect(self.endCapture)
        self.end_button = QtWidgets.QPushButton('Stop')

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.end_button)
        vbox.addWidget(self.quit_button)
        self.setLayout(vbox)
        self.setWindowTitle('Control Panel')
        self.setGeometry(100,100,200,200)
        self.show()

    def startCapture(self):
        if not self.capture:
            self.capture = QtCapture(0)
            self.end_button.clicked.connect(self.capture.stop)
            # self.capture.setFPS(1)
            self.capture.setParent(self)
            self.capture.setWindowFlags(QtCore.Qt.Tool)
        self.capture.start()
        self.capture.show()

    def endCapture(self):
        self.capture.deleteLater()
        self.capture = None


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = ControlWindow()
    sys.exit(app.exec_())



