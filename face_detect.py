import cv2
#imagePath = "C:\\Users\\venka_000\\Documents\\GitHub\\OpencvWorkspace\\messi.jpg"
#cascPath = 'haarcascade_frontalface_default.xml'

# creating cascade
faceCascade = cv2.CascadeClassifier("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml")

print( faceCascade.empty() )
#read the image
image = cv2.imread('C:\\Users\\venka_000\\Documents\\GitHub\\OpencvWorkspace\\FaceDetection\\s1.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Detetct faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5
)

print "Found {0} faces!".format(len(faces)) 

#drawing rectangle around faces
for(x, y, w, h) in faces:
    cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0),2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
    