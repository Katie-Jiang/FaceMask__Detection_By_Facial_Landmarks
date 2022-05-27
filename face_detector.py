import os
import time
import sys
import cv2
import dlib
import numpy as np
import tensorflow as tf

# cascPath = sys.argv[1]
TYPE = {0:"No Mask on Face", 1:"Nose not Covered", 2:"Mouth not Covered", 3:"Correctly Wear Mask"}

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


cascPath = os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

tf.executing_eagerly()
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session())
model_mouth = tf.keras.models.load_model('model_r_mouth', custom_objects={"tf": tf})
model_nose = tf.keras.models.load_model('model_r_nose', custom_objects={"tf": tf})

prev_faces = []
prev_type = "Waiting"
prev_outline = []
prev_outline_nose = []
prev_outline_mouth = []

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=7,
        minSize=(100,100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        faces = prev_faces
    else:
        prev_faces = faces

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # type(face) == <class 'numpy.ndarray'>
        face_box = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        image = frame[y:y+h, x:x+w]
        image = cv2.resize(image, (500, 500))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if len(detector(image)) != 0:
            rect = detector(image)[0]
            sp = predictor(image, rect)
            landmarks = np.array([[p.x, p.y] for p in sp.parts()])
            outline = landmarks[[*range(67,29,-1)]]
            outline_nose = landmarks[[*range(36,29,-1)]]
            outline_mouth = landmarks[[*range(67,49,-1)]]

            prev_outline = outline
            prev_outline_nose = outline_nose
            prev_outline_mouth = outline_mouth
        else:
            outline = prev_outline
            outline_nose = prev_outline_nose
            outline_mouth = prev_outline_mouth

        if len(outline) != 0 and len(outline_nose) != 0 and len(outline_mouth) != 0:
            # get the nose and mouth area
            crop_nose = image[min(outline_nose[:,1]):max(outline_nose[:,1]), min(outline_nose[:,0]):max(outline[:,0])]
            crop_mouth = image[min(outline_mouth[:, 1]):max(outline_mouth[:, 1]), min(outline_mouth[:, 0]):max(outline[:, 0])]
            # crop_nose_mouth = image[min(outline[:,1]):max(outline[:,1]), min(outline[:,0]):max(outline[:,0])]
            
            crop_nose = cv2.resize(crop_nose, (150,150))
            crop_mouth = cv2.resize(crop_mouth, (150,150))
            # crop_nose_mouth = cv2.resize(crop_nose_mouth, (150,150))

            # cv2.imwrite("crop_nose.png", crop_nose)
            # cv2.imwrite("crop_mouth.png", crop_mouth)
            # cv2.imwrite("crop_nose_mouth.png", crop_nose_mouth)

            label_m, prob_m = model_mouth.predict(np.expand_dims(crop_mouth, 0))
            # print(('MOUTH Predicted labels: %d' % label_m[0]))
            # print(('Probability: %f' % prob_m[0]))
            label_n, prob_n = model_nose.predict(np.expand_dims(crop_nose, 0))
            # print(('NOSE Predicted labels: %d' % label_n[0]))
            # print(('Probability: %f' % prob_n[0]))

            if label_m == 0 and label_n == 0:
                status = TYPE[3]
            elif label_m == 1 and label_n == 1:
                status = TYPE[0]
            elif label_n:
                status = TYPE[1]
            elif label_m:
                status = TYPE[2]

            prev_type = status
            if status == TYPE[3]:
                cv2.putText(face_box, status, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            else:
                cv2.putText(face_box, status, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        else:
            # type <- model
            # type = TYPE[0]
            if prev_type == TYPE[3]:
                cv2.putText(face_box, prev_type, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            else:
                cv2.putText(face_box, prev_type, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    time.sleep(0.08)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()