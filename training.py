import cv2
import numpy as np
from labelImages import getLabelAndImages
recognizer = cv2.face.LBPHFaceRecognizer_create()
# print(recognizer)
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faces, Id = getLabelAndImages("TrainingImages")
recognizer.train(faces, np.array(Id))
recognizer.save("TrainingImageLabel\Trainer.yml")

print("Model Trained Successfully")