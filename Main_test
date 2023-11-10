from tensorflow import keras
import cv2 as cv

lista = ['Bad','Good']

model = keras.models.load_model('CNN_MASK.h5')
cap = cv.VideoCapture(0, cv.CAP_DSHOW)

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(cascPath)


if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60,60))
    
    #por cada cara detectada pintar un cuadro
    for (x, y, w, h) in faces:
        frame = cv.rectangle(frame, (x, y), (x+w, y+h), (0, 210, 255), 10)
        face = gray[y:y+h,x:x+w]
        
        if face.shape[0]>50 and face.shape[1]>50:
            binary = cv.resize(face, (50,50))
            patron = binary.reshape(1,50,50,1)
        
            pred = model.predict(patron);
            lab_pred = lista[(pred[0,0]>0.5)*1]
            
            frame = cv.putText(frame,lab_pred, (x,y), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
        
            cv.imshow('face', face)

    
    # Display the resulting frame
    cv.imshow('frame', frame)
    
    if cv.waitKey(1) == ord('q'):
        break
    
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
