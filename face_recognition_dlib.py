import cv2
import dlib

hair_image = cv2.imread('hair1.png', cv2.IMREAD_UNCHANGED)

if hair_image is None:
    print("Error: Could not read the image file")
    exit()

detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        
        face_region = frame[y:y+h, x:x+w]
        
        resized_hair = cv2.resize(hair_image, (w, h))
        
        for c in range(0, 3):
            if hair_image.shape[2] == 4:
                face_region[:,:,c] = resized_hair[:,:,c] * (resized_hair[:,:,3] / 255.0) + frame[y:y+h, x:x+w][:,:,c] * (1.0 - resized_hair[:,:,3] / 255.0)
            else:
                face_region[:,:,c] = resized_hair[:,:,c]  
    
    cv2.imshow('Hair Style Filter', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
