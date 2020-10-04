import cv2


def generate_dataset(img, name):

    cv2.imwrite("faces/"+str(name)+".jpg", img)

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []

    for (x, y, w, h) in (features):
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        '''id, _ = clf.predict(gray_img[y:y+h, x:x+w])
        if id==1:
            cv2.putText(img, "Buris", (x, y-4), cv2.FONT_HERSHEY_TRIPLEX, 0.8, color, 1, cv2.LINE_AA)'''

        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_TRIPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return  coords, img

def recognize(img, clf, faceCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0)}

    coords = draw_boundary(img, faceCascade, 1.1, 10, color["red"], "Face", clf)
    return coords

def detect(img, faceCascade):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0)}

    coords, img = draw_boundary(img, faceCascade, 1.2, 10, color['green'], "Face")

    if len(coords)==4:
        roi_img = img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]]
        name = input("What is your name")
        generate_dataset(roi_img, name)

   # ^^^crop image to the size of the face then extract each frame into .jpg into data folder ready to be train and assign ID^^
    '''if len(coords) == 0:
        print("No faces found")


    else:
        print(coords)


        print("Number of faces detected: " + str(coords[0]))'''

    return  img


faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#clf = cv2.face.LBPHFaceRecognizer_create()
#clf.read("train.yml")
video_capture = cv2.VideoCapture(0)



#path = 'E:\Snek_project\Face_recog\data'
#spec = os.mkdir(os.path.join(path, name))




while True:
    _, img = video_capture.read()
    img = detect (img, faceCascade)
    #img = recognize(img, clf, faceCascade)
    #img = generate_dataset(img, id, img_id)
    cv2.imshow("face detection", img)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()