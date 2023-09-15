import cv2
import pandas as pd

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("TrainingImageLabel\Trainer.yml")
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

df = pd.read_csv(r"UserDetails\UserDetails.csv")
# print(df)
url = "http://10.109.24.42:4747/video"
# url = "http://localhost:4747/video"
cam = cv2.VideoCapture(url)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    print(faces)
    for(x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
        Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
        print("id-----------> ",Id)
        print("conf-------> ",conf)
        if(conf < 80):
            aa = df.loc[df['Id'] == Id]['Name'].values
            print(aa)
            tt = str(Id)+"-"+aa
        else:
            Id = 'Unknown'
            tt = str(Id)
        cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
    cv2.imshow('im', im)
    if (cv2.waitKey(1) == ord('q')):
        break
cam.release()
cv2.destroyAllWindows()
