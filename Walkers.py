import cv2

cap = cv2.VideoCapture('walking.avi') 
frame = cap.read()

gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

bodies = body_classifier.detectMultiScale(gray,1.2,3)
print(bodies)

for (x,y,w,h) in bodies:
       cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
             
cv2.imshow('img',frame)

  
# After the loop release the cap object
cap.release()

# Destroy all the windows
cv2.destroyAllWindows()