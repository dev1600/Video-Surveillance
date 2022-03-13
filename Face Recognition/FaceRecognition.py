import cv2
import numpy as np
import os

subjects=["Intruder",'Devansh','Harsh']

def detect_face(img):
    # convert the test image to gray scale as opencv 
    # face detector expcts gray images
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #load OpenCV face detector,I am using LBP which is fast
    # there is also a more accurate but slow Haar classifier
    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # let's detect multiscale images(some images may be closer)
    faces = face_cascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )
    
    # if no faces are detected then return original image
    if(len(faces)==0):
        return None,None
    
    # Assuming only one face is present
    (x,y,w,h)=faces[0]
    
    #returns only face part of the image 
    return gray[y:y+w,x:x+h],faces[0]

def prepare_training_data(data_folder_path):
    
    # get the directories (one directory for each subject)
    # os.setwd()
    # os.chdir('/home/dev16/Project/Face Recognition')
    # data_folder_path=os.getcwd()+data_folder_path
    dirs=os.listdir(data_folder_path)
    
    # list to hold all subject faces
    faces=[]
    
    # list to hold labels for all subjects
    labels=[]
    
    # Going through the directory and read images
    for dir_name in dirs:
        
        if not dir_name.startswith('s'):
            continue
        
        # Extracting label number of subject
        label=int(dir_name.replace('s',''))
        
        #build path of directory containing images for current
        # object 
        subject_dir_path=data_folder_path+'/'+dir_name
        
        # get the images name that are inside the given subject
        subject_images_names=os.listdir(subject_dir_path)
        
        #go through each image name read image
        # detect face and add face to list of faces
        for image_name in subject_images_names:
            
            if(image_name.startswith('.')):
                continue
            
            #build image path
            #sample image path=training-data/s1/1.pgm
            image_path=subject_dir_path+'/'+image_name
            
            # read image
            image=cv2.imread(image_path)
            
            cv2.imshow("Training on image...",cv2.resize(image,(400,500)))
            cv2.waitKey(30)
            
            # detect face
            face,rect=detect_face(image)
            
            # Ignore faces that are not detected
            if face is not None:
                # add face to list of faces
                faces.append(face)
                # add label for this face
                labels.append(label)
                
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
        
    return faces,labels
    
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)

    if(face is None):
        draw_text(img,"No Face detected", 250, 200)
        return img

    # print("Hello sexi")
    #predict the image using our face recognizer 
    label, confidence = face_recognizer.predict(face)
    #get name of respective label returned by face recognizer
    label_text = subjects[label]
    
    #draw a rectangle around face detected
    draw_rectangle(img, rect)
    #draw name of predicted person
    draw_text(img, f'{label_text} {confidence}', rect[0], rect[1]-5)
    
    return img

print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))      


face_recognizer =  cv2.face.LBPHFaceRecognizer_create()
# face_recognizer = cv2.face.EigenFaceRecognizer_create()
# face_recognizer =  cv2.createLBPHFaceRecognizer()

face_recognizer.train(faces, np.array(labels))

print("Predicting images...")

#load test images
test_img1 = cv2.imread("test-data/test3.jpg")

# test_img2 = cv2.imread("test-data/test2.jpg")
# cv2.imshow("RaN",cv2.resize(test_img1,(400,500)))

#perform a prediction
predicted_img1 = predict(test_img1)
# predicted_img2 = predict(test_img2)
print("Prediction complete")

#display both images
cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
# cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()