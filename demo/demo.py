import cv2 as cv
import mediapipe as mp
import torch
import numpy as np
import matplotlib.colors as mcolors
from model_trust import TrustAwareBayesianMobNet
import albumentations as A
import os

# Load the hand pose estimation model
model = TrustAwareBayesianMobNet()
model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'demo', 'bnn.pt')))
model.eval()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize Video Capture
cap = cv.VideoCapture(0)  # 0 for the default webcam

def renderPose(img, uv):
    colors = ['blue', 'orange', 'lime', 'yellow', 'cyan', 'black']
    connections = [[0,1], [2,0], [3,2], [4,5],
                   [6,4], [7,6], [6,7], [8,9],
                   [11,10], [10,8],[12,13], [14,12],
                   [15,14],[16,17], [18,16], [19,18],
                   [20,1],[20,5], [20,9], [20,13], [20, 17]]
    for c in connections:
        a, b = uv[c[0]].astype(int), uv[c[1]].astype(int)
        img = cv.line(img, tuple(a), tuple(b), (0, 0, 255), 2)
    iterr = -1
    for ind, point in enumerate(uv):
        if ind % 4 == 0:
            iterr += 1
        rgb = tuple(map(lambda x: int(x * 255), mcolors.to_rgb(colors[iterr])))
        img = cv.circle(img, tuple(point.astype(int)), 5, rgb, -1)
    return img

def predict_(model, img, bbox, IMG_SIZE=224):
    x1, y1, x2, y2 = list(map(int, bbox))
    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    if x2 <= x1: x2 = x1 + 1
    if y2 <= y1: y2 = y1 + 1
    transformer = A.Compose([
        A.Crop(x_min=x1, y_min=y1, x_max=x2, y_max=y2, always_apply=True, p=1.0),
        A.Resize(IMG_SIZE, IMG_SIZE)])
    img_transformed = transformer(image=img)['image']
    img_transformed = (img_transformed / 255).astype(np.float32)
    img_tensor = torch.from_numpy(img_transformed).permute(-1, 0, 1)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    model.eval()
    with torch.no_grad():
        preds = torch.squeeze(model(img_tensor))
    keypoints = preds.reshape(-1, 2).numpy()
    keypoints[:, 0] = keypoints[:, 0] * (x2 - x1) + x1
    keypoints[:, 1] = keypoints[:, 1] * (y2 - y1) + y1
    return renderPose(img, keypoints)

while cap.isOpened():
    try:
        success, frame = cap.read()
        if not success:
            continue  # Skip to the next frame if reading fails

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                h, w, _ = frame.shape
                x_min, y_min = min([lm.x for lm in hand_landmarks.landmark]) * w, min([lm.y for lm in hand_landmarks.landmark]) * h
                x_max, y_max = max([lm.x for lm in hand_landmarks.landmark]) * w, max([lm.y for lm in hand_landmarks.landmark]) * h
                bbox = [x_min, y_min, x_max, y_max]
                img_with_pose = predict_(model, frame, bbox)
                frame = img_with_pose
        cv.imshow('Hand Pose', frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"An error occurred: {e}")
        continue

cap.release()
cv.destroyAllWindows()