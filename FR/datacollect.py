import cv2
import os
import face_recognition as fr
import numpy as np

def get_encoded_faces():
    encoded = {}
    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                face = cv2.resize(face, (0,0), fx=0.5, fy=0.5)
                encoding = fr.face_encodings(face)
                if encoding:
                    encoded[f.split(".")[0]] = encoding[0]
    return encoded      

def classify_face(img,faces_encoded,known_face_names):
    face_locations = fr.face_locations(img)
    unknown_face_encodings = fr.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        matches = fr.compare_faces(faces_encoded, face_encoding)
        name = "???"

        face_distances = fr.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)
        cv2.rectangle(img, (left-20, bottom-15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, name, (left-20, bottom+15), font, 1.0, (255, 255, 255), 2)

    return img

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

faces = get_encoded_faces()
faces_encoded = list(faces.values())
known_face_names = list(faces.keys())

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = classify_face(frame,faces_encoded,known_face_names)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
