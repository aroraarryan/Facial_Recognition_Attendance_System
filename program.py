import face_recognition
import cv2
import numpy as np
import os
import csv
from datetime import datetime

# Taking input from the webcam
video_capture = cv2.VideoCapture(0)

# Creating a variable for the known faces

bill_image = face_recognition.load_image_file("photos/Bill.jpg")
bill_encoding = face_recognition.face_encodings(bill_image)[0]

narendra_image = face_recognition.load_image_file("photos/Narendra.jpeg")
narendra_encoding = face_recognition.face_encodings(narendra_image)[0]

ratan_image = face_recognition.load_image_file("photos/Ratan.jpeg")
ratan_encoding = face_recognition.face_encodings(ratan_image)[0]

steve_image = face_recognition.load_image_file("photos/Steve.jpg")
steve_encoding = face_recognition.face_encodings(steve_image)[0]

known_face_encoding = [
    bill_encoding,
    narendra_encoding,
    ratan_encoding,
    steve_encoding
]

known_faces_names = [
    "Bill Gates",
    "Narendra Modi",
    "Ratan Tata",
    "Steve Jobs"
]

students = known_faces_names.copy()


face_locations = []     # collects face coordinates
face_encodings = []     # collects raw data
face_names = []         # collects names if present in the list
s=True            

# getting the current date and time

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# creating a csv file

f= open(current_date+'.csv','w+',newline='')
lnwriter = csv.writer(f)


while True:             # infinte loop using while loop
    _,frame = video_capture.read()    # extracting the video data
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)      # decreasing the size of input    
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])   # converting the frame into rgb
    
    # using face_recognition to detect faces

    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)       # if there is a face in the frame or not
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)        
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)    # comparing the current face with the known faces
            best_match_index = np.argmin(face_distance)     # getting the best probability of a face using numpy
            if matches[best_match_index]:
                name= known_faces_names[best_match_index]   # getting the name of the person
            
            
            # Adding the data to the csv file
            face_names.append(name)
            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,100)
                fontScale              = 1.5
                fontColor              = (255,0,0)
                thickness              = 3
                lineType               = 2
 
                cv2.putText(frame,name+' Present', 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
                if name in students:
                    students.remove(name)     # So that we Don't repeat the name
                    print(students)
                    current_time= now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])
    
    cv2.imshow("attendence system",frame)    # to show the result
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()