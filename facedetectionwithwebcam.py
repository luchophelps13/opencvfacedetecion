import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read() #get webcam img
    cv2.setWindowTitle("img", "")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grayscale (no need for color)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #detect face based on haar cascades (rectangle shapes)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2) #draw rectangle on face
        cv2.setWindowTitle('img', "Face Detected")
        roi_gray = gray[y:y+h, x:x+w] #roi means region of interest (draws on the gray img)
        roi_color = img[y:y+h, x:x+w] #draws on the color img
        
        eyes = eye_cascade.detectMultiScale(roi_gray) #find eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) #draw box
            cv2.setWindowTitle('img', "Face & Eyes Detected")

    cv2.imshow('img', img) #show img
    k = cv2.waitKey(30) & 0xff #escape key
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
