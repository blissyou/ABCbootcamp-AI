import cv2
import numpy as np
model = "YOLO3.weights"
config = "YOLO3.cfg"
class_labels = "coco.names"
confThreshold = 0.5
nmsThreshold = 0.4

# 테스트 이미지들
img_files = ['bicycle.jpg','car.jpg','dog.jpg',
             'kite.jpg','person.jpg','sheep.jpg',
             ]
# 모델 읽어오기
net = cv2.dnn.readNet(model=model, config=config)

with open(file=class_labels,mode='rt') as f:
    classes = f.read().split('\n')

print(classes)

# 랜덤하게 컬러를 화면에 보여준다 (텍슽,윈도우)
colors = np.random.uniform(low=0,high=255,size=(len(classes),3))
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1]for i in net.getUnconnectedOutLayers()]
for f in img_files:
    img=cv2.imread(f)
    blob = cv2.dnn.blobFromImage(image=img,scalefactor= 1/255.,size=(416,416),swapRB=True,crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    (h, w) = img.shape[:2]
    class_ids = list()
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confThreshold:
                cx = int(detection[0] * w)
                cy = int(detection[1] * h)
                bw = int(detection[2] * w)
                bh = int(detection[3] * h)

                sx =int(cx-bw/2)
                sy = int(cy-bh/2)

                boxes.append([sx,sy,bw,bh])
                confidences.append(float(confidence))
                print(class_id(float(confidence)))
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold,nmsThreshold)
    for i in indices:
        (sx,sy,bw,bh)=boxes[i]
        label = f'{classes[class_ids[i]]} : {confidences[i]:.2}'
        color = colors[class_ids[i]]
        cv2.rectangle(img,(sx,sy),(bw,bh),color,2)
        cv2.putText(img,label,(sx,sy-10),cv2.FONT_HERSHEY_PLAIN,0.7,color,2,cv2.LINE_AA)
    (t,_) = net.getPerfProfile()
    label = "Inference time : %2.f ms" % (t*1000.0/cv2.getTickFrequency())
    cv2.putText(img,label,(10,30),
                cv2.FONT_HERSHEY_PLAIN,0.7,(0,0,255),1,cv2.LINE_AA)
    cv2.imshow('img',img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
