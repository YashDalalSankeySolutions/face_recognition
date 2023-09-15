import cv2
import os
import time
import csv
name = str(input("enter Name:- "))
Id = str(input("enter id:- "))

url = "http://10.109.24.42:4747/video"
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)
count = 0
# ret,frame = video.read()
while count<1:

    ret,frame = video.read()
    print("===========>",frame)
    if not ret:
        print("Failed to capture image.")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = facedetect.detectMultiScale(frame, 1.3, 5)
    # faces = facedetect.detectMultiScale(gray,
    #                                      scaleFactor=1.1,
    #                                      minNeighbors=5,
    #                                      minSize=(60, 60),
    #                                      flags=cv2.CASCADE_SCALE_IMAGE)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    print("Capturing Image",faces)
    print("count: ",count)
    # time.sleep(3)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        count= count+1
        print("hhh")
        # image_name = f"{name}_{count + 1}.jpg"
        image_name = f"{name}.{Id}.{count}.jpg"
        face_image = frame[y:y+h,x:x+w]                          
        # print("face_image",face_image)
        # cv2.imwrite('image/'+image_name, face_image)
        cv2.imwrite('images/'+image_name, face_image)
        print(f"Image {count} captured and saved as '{image_name}'.")
    
    cv2.imshow("image_name",frame)
    # if cv2.waitKey(100) & 0xFF == ord('q'):
    #     break
    time.sleep(1)
video.release()  
cv2.destroyAllWindows()
# row = [Id, name]
# with open(r'UserDetails\UserDetails.csv', 'a+') as csvFile:
#     writer = csv.writer(csvFile)
#     # Entry of the row in csv file
#     writer.writerow(row)
# csvFile.close()
# res = "Images Saved for ID : " + Id + " Name : " + name
# print(res)
    

