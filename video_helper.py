import cv2
from deepface import DeepFace

def get_frame_number(time_stamp, fps):
    return int(time_stamp * fps)


def seek_to_frame(cap, frame_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    #return cap.grab()

def get_frame(video_path, timestamp):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = get_frame_number(timestamp, fps)
    seek_to_frame(cap, frame_number)
    ret, frame = cap.read()
    cap.release()
    return frame


def display_image_cv2(image):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


def draw_bounding_boxes(image, bounding_boxes):
    print(f"Debug : bounding_boxes {bounding_boxes}")
    for bounding_box in bounding_boxes:
        x, y, w, h = bounding_box
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image


def detect_faces(image_path):
    faces = DeepFace.extract_faces(image_path, enforce_detection=False)
    return faces


def get_all_frames(video_path, start_timestamp=0, interval=30):
    #start from start_frame
    cap = cv2.VideoCapture(video_path)
    print(f"Debug : FPS {cap.get(cv2.CAP_PROP_FPS)}")
    start_frame = get_frame_number(start_timestamp, cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    print(f"Debug in get_all_frames : current_pos {current_pos}")
    if current_pos != start_frame:
        raise ValueError(f"Failed to set start frame. Requested: {start_frame}, Got: {current_pos}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        print(f"Debug in get_all_frames : frame_number {frame_number}")
        if frame_number % interval == 0:
            yield frame_number, frame

        for _ in range(interval- 1):
            ret = cap.grab()
            if not ret:
                break
        
    cap.release()