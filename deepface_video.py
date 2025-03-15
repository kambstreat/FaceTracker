from deepface import DeepFace
from video_helper import get_all_frames, draw_bounding_boxes, display_image_cv2
import matplotlib.pyplot as plt
import cv2
from image_database import get_embedding

embedding1 = get_embedding()

#match the embedding to the video

for frame_number, frame in get_all_frames('video1.mp4', start_timestamp=30, interval=30):
    face1 = DeepFace.extract_faces(frame, enforce_detection=False)
    confidence = face1[0]["confidence"]
    #if confidence < 0.9:
    #    print(f"Debug : Frame {frame_number}; confidence {confidence}")
    #    continue
    
    embedding2 = DeepFace.represent(frame)[0]['embedding']


    print(f" embedding 1: {embedding1[:100]}")
    print(f" embedding 2: {embedding2[:100]}")

    distance = DeepFace.verify(img1_path=embedding1, img2_path=embedding2)

    print(f"Debug : Frame {frame_number}; distance {distance}")

    ## comment for loo
    #for face in face1:
    #    display_image_cv2(face["face"])

        
    #    x = face["facial_area"]["x"]
    #    y = face["facial_area"]["y"]
    #    w = face["facial_area"]["w"]
    #    h = face["facial_area"]["h"]

    #    print(f"x: {x}, y: {y}, w: {w}, h: {h}")

    #    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #    display_image_cv2(frame)


    break