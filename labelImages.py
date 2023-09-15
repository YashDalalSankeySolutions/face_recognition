import os
from PIL import Image
import numpy as np

def getLabelAndImages(path):
    imagePaths  = [os.path.join(path,f) for f in os.listdir(path)]
    # print(imagePaths)
    faces=[]
    Id =[]

    for imagePath in imagePaths:
        # print("imagepath----->",imagePath)
        pilImage = Image.open(imagePath).convert('L')
        # print("PilImage-----> ",pilImage)
        imageNp = np.array(pilImage,'uint8')
        # print("imageNp------->",imageNp)
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        # print("id-----> ",id)
        faces.append(imageNp)
        Id.append(id)
    # print("faces-----> ",faces)
    print("Id-----> ",Id)
    return faces,Id

# getLabelAndImages("TrainingImages")

