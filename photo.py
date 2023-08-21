import yolov5

# load pretrained model
# model = yolov5.load('yolov5s.pt')

# or load custom model
model = yolov5.load('best.pt')
  
# set model parameters
model.conf = 0.1  # NMS confidence threshold
model.iou = 0.1  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# set image
img = 'DHT11-Temperature-And-Humidity-Sensor-srkelectronics.in_.jpg'

# perform inference
results = model(img)

# inference with larger input size
results = model(img, size=1280)

# inference with test time augmentation
results = model(img, augment=True)

# parse results
predictions = results.pred[0]
boxes = predictions[:, :4] # x1, y1, x2, y2
scores = predictions[:, 4]
categories = predictions[:, 5]

# show detection bounding boxes on image
results.show()


