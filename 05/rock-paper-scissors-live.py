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

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('recorded_video.mp4', fourcc, 20.0, (int(camera.get(3)), int(camera.get(4))))

while True:
    ret, image = camera.read()

    if not ret:
        print("Failed to grab frame")
        break

    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    image_for_prediction = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_for_prediction = (image_for_prediction / 127.5) - 1
    prediction = model.predict(image_for_prediction)

    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    text = f"Class: {class_name}, Confidence: {np.round(confidence_score * 100)}%"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Webcam Image", image)
    out.write(image)

    keyboard_input = cv2.waitKey(1)

    if keyboard_input == 27: 
        break
    elif keyboard_input == ord('q'):
        print("Recording stopped by user.")
        break

camera.release()
out.release()
cv2.destroyAllWindows()
