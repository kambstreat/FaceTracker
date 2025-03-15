from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
import torch
from torch.utils.data import DataLoader
from torchvision import datasets


from PIL import Image, ImageDraw
from torchcodec.decoders import SimpleVideoDecoder
import cv2
from torchvision import transforms
import numpy as np
from image_database import get_embedding, get_image_from_wiki
from deepface import DeepFace

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_face_embedding(face_image, model=None, device='cpu'):
    """
    Get face embedding using InceptionResnetV1
    
    Args:
        face_image: PIL Image or tensor of face
        model: InceptionResnetV1 model (if None, will create new)
        device: 'cuda' or 'cpu'
    """
    # Initialize model if not provided
    if model is None:
        model = InceptionResnetV1(pretrained='vggface2').eval()
        model = model.to(device)
    
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((160, 160)),  # InceptionResnetV1 expects 160x160
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Convert to tensor if needed
    if not isinstance(face_image, torch.Tensor):
        face_image = transform(face_image)
    
    # Add batch dimension if needed
    if face_image.dim() == 3:
        face_image = face_image.unsqueeze(0)
    
    # Move to device and get embedding
    face_image = face_image.to(device)
    with torch.no_grad():
        embedding = model(face_image)
    
    return embedding.cpu()



#JP narayan : pageid = 18404696
#Brahmanandam : pageid = 2298012
def get_wiki_embedding(pageid):
    img = get_image_from_wiki(pageid)
    boxes = detect_faces_from_frame(img)
    face = mtcnn.extract(img, boxes, None)
    display_frame_opencv(face)
    embedding = get_face_embedding(face)
    return embedding


def detect_faces_from_frame(frame):
    boxes, _ = mtcnn.detect(frame)
    return boxes

def display_frame(frame, boxes=None):
    transform = transforms.ToPILImage()
    frame = transform(frame)
    draw = ImageDraw.Draw(frame)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle((x1, y1, x2, y2), width=5)

    frame.show()
    #proceed only if the user presses the enter key
    print("Press enter to continue")
    cv2.waitKey(0)
    return frame


def display_frame_opencv(frame, boxes=None):
    # Convert tensor to numpy array for OpenCV
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
        if frame.shape[0] == 3:  # CHW format
            frame = frame.transpose(1, 2, 0)  # Convert to HWC
    
    # Convert to uint8 if needed
    if frame.dtype == np.float32:
        frame = (frame * 255).astype(np.uint8)
    
    # Draw boxes if provided
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Display frame
    cv2.imshow('Frame', frame)
    
    # Wait for key press and close window
    key = cv2.waitKey(0)  # 0 means wait indefinitely
    cv2.destroyWindow('Frame')
    
    return frame, key

def read_frames_from_video(video_path):
    decoder = SimpleVideoDecoder(video_path)
    counter = 0
    for frame in decoder[750:-1:752]:
        display_frame_opencv(frame)
        permuted_frame = frame.permute(1,2,0)
        boxes = detect_faces_from_frame(permuted_frame)
        for box in boxes:
            box_np = np.array(box).reshape(1, -1)
            face = mtcnn.extract(permuted_frame, box_np, None)
            embedding = get_face_embedding(face)
            result = DeepFace.verify(embedding.numpy().tolist()[0], get_wiki_embedding(2298012).numpy().tolist()[0], model_name='Facenet512')
            print("--------------------------------")
            print(result)
            print("--------------------------------")
        display_frame_opencv(frame, boxes)
        break






if __name__ == '__main__':
    video_path = "video1.mp4"
    read_frames_from_video(video_path)
    #embedding = get_jp_embedding()
    #print(embedding)

