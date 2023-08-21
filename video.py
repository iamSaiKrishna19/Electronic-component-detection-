import yolov5
import cv2
import numpy as np

# Load the YOLOv5 model
model = yolov5.load('best.pt')

# Set model parameters
model.nms_conf = 0.65  # NMS confidence threshold
model.nms_iou = 0.85  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_detections = 1000  # maximum number of detections per image

# Open the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    
    # Perform inference on the frame
    results = model(frame)

    results = model(frame, augment=True)
    # Parse the results
    predictions = results.pred[0].detach().cpu().numpy()
    boxes = predictions[:, :4]  # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    # Show detection bounding boxes on the frame
    for box, score, category in zip(boxes, scores, categories):
        x1, y1, x2, y2 = np.round(box).astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow('Frame', frame)
    

    # Check for 'q' key to exit
    if cv2.waitKey(15) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
