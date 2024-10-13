from keras.models import load_model 
import cv2  
import numpy as np

np.set_printoptions(suppress=True)

model = load_model("keras_Model.h5", compile=False)

class_names = open("labels.txt", "r").readlines()

camera = cv2.VideoCapture(0)

window_width = 700
window_height = 500

cv2.namedWindow("Webcam Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webcam Image", window_width, window_height)

while True:
    ret, image = camera.read()

    if not ret:
        print("Failed to grab frame")
        break

    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imshow("Webcam Image", image)

    image_for_prediction = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_for_prediction = (image_for_prediction / 127.5) - 1
    prediction = model.predict(image_for_prediction)

    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    
    keyboard_input = cv2.waitKey(1)

    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
