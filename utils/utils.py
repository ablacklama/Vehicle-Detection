import numpy as np
import cv2

class bb:
    def __init__(self):
        self.x, self.y, self.h, self.w, self.c, self.prob, self.x1, self.y1, self.x2, self.y2\
            = None, None, None, None, None, None, None, None, None, None

#change data into propper size for Yolonet. Also can crop image with shavedim=(ymin,ymax,xmin,xmax)
def preprocess(img, shave=False, shavedim=(350,500, 500,1000)):
    #if the image is to be cropped make sure the values are sane first then reduce the image down to the new dimentions
    if shave:
        if(shavedim[0] < 0):
            shavedim[0] = 0
        if (shavedim[1] > img.shape[0]):
            shavedim[1] = img.shape[0]
        if (shavedim[2] < 0):
            shavedim[2] = 0
        if (shavedim[3] > img.shape[1]):
            shavedim[3] = img.shape[1]
        img = img[shavedim[0]:shavedim[1],shavedim[2]:shavedim[3]]
    sizexy = [img.shape[1], img.shape[0]]

    #get the appropriate padding on the image to make it square
    padhw = [0,0]
    if(sizexy[0] > sizexy[1]):
        dif = sizexy[0] - sizexy[1]
        border = cv2.copyMakeBorder(img, int(dif/2), int(dif/2), 0, 0, cv2.BORDER_CONSTANT, value=[200, 200, 200])
        padhw[0] = int(((dif/2)/border.shape[0]) * 448)

    elif (sizexy[1] > sizexy[0]):
        dif = sizexy[1] - sizexy[0]
        border = cv2.copyMakeBorder(img, 0, 0, int(dif / 2), int(dif / 2), cv2.BORDER_CONSTANT, value=[200, 200, 200])
        padhw[1] = int(((dif / 2) / border.shape[1]) * 448)
    else:
        border = img

    #resize the image to fit the 448,448 input that yolo requires
    resized = cv2.resize(border, (448, 448))

    #yolo requires the image to be fed in by (channel, y,x). Transpose to match that.
    transposed = np.transpose(resized, [2, 0, 1])
    return transposed, padhw, shavedim, resized

#read weights from file and load them into the model
def load_weights(model,yolo_weight_file):
    weights = np.fromfile(yolo_weight_file,np.float32)
    weights = weights[4:]
    
    index = 0
    for layer in model.layers:
        shape = [w.shape for w in layer.get_weights()]
        if shape != []:
            shape_kernal,shape_bias = shape
            bias = weights[index:index+np.prod(shape_bias)].reshape(shape_bias)
            index += np.prod(shape_bias)
            kernal = weights[index:index+np.prod(shape_kernal)].reshape(shape_kernal)
            index += np.prod(shape_kernal)
            layer.set_weights([kernal,bias])


#change yolo output into box values that corrospond to the original image.
def process_output(yolo_output, threshold=0.2, padhw=(98,0), shaved=False, shavedim=(350,500, 500,1000)):
    # Class label for car in the dataset
    car_class = 6
    boxes = []
    S = 7
    B = 2
    C = 20
    SS = S * S  # num yolo grid cells
    prob_size = SS * C  # num class probabilities
    conf_size = SS * B  # num confidences, 2 per grid cell

    probs = yolo_output[0:prob_size]  # seperate probability array
    confidences = yolo_output[prob_size:prob_size + conf_size]  # seperate confidence array
    yolo_boxes = yolo_output[prob_size + conf_size:]  # seperate coordinates

    # reshape arrays so that each cell in the yolo grid is a seperate array containing the cells properties
    probs = np.reshape(probs, (SS, C))
    confs = np.reshape(confidences, (SS, B))
    yolo_boxes = np.reshape(yolo_boxes, (SS, B, 4))

    # itterate through grid and then boxes in each cell
    gridn = 0
    for gridy in range(S):
        for gridx in range(S):
            for index1 in range(B):

                box = bb()
                box.c = confs[gridn, index1]
                p = probs[gridn, :] * box.c

                if (p[car_class] >= threshold):

                    #find pixel values of xywh in current 448,448 image
                    box.x = yolo_boxes[gridn, index1, 0] * (448 / 7) + (gridx * (448 / 7))
                    box.y = yolo_boxes[gridn, index1, 1] * (448 / 7) + (gridy * (448 / 7))
                    box.w = yolo_boxes[gridn, index1, 2] / 2 * 448
                    box.h = yolo_boxes[gridn, index1, 3] / 2 * 448

                    # scale y to match current image ratio without border padding
                    box.y = box.y - padhw[0]
                    box.x = box.x - padhw[1]

                    # scale boxes pixel values to original image (values still wrong if image was shaved)
                    nopadh = 448 - (padhw[0]*2)
                    nopadw = 448 - (padhw[1]*2)
                    if shaved:
                        x_scale = (shavedim[3]-shavedim[2]) / nopadw
                        y_scale = (shavedim[1]- shavedim[0])/nopadh
                    else:
                        x_scale = 1280 / nopadw
                        y_scale = 720 / nopadh
                    box.y = box.y * y_scale
                    box.w = box.w * x_scale
                    box.x = box.x * x_scale
                    box.h = box.h * y_scale

                    #add shaved pixel amounts to coordinates to adjust them back to the original image.
                    if shaved:
                        box.y += shavedim[0]
                        box.x += shavedim[2]

                    box.prob = p[car_class]
                    boxes.append(box)
            gridn += 1
    # sort in decending order by confidence
    boxes.sort(key=lambda box: box.prob, reverse=True)
    return boxes


#remove boxes that are most likely duplacates
def remove_duplicates(boxes, img):
    h, w, _ = img.shape
    for box in boxes:
        box.x1 = int(box.x - (box.w / 2))
        box.x2 = int(box.x + (box.w / 2))
        box.y1 = int(box.y - (box.h / 2))
        box.y2 = int(box.y + (box.h / 2))

        # set boxes to be within the border of the picture
        if box.x2 > img.shape[1] - 3:
            box.x2 = img.shape[1] - 3
        if box.y2 > img.shape[0] - 3:
            box.y2 = img.shape[0] - 3

        if box.x1 < 3:
            box.x1 = 3
        if box.y1 < 3:
            box.y1 = 3

    # remove boxes that are to similar to a box with a better confidence score
    for index1 in range(len(boxes)):
        box1 = boxes[index1]
        if box1.prob == 0: continue
        for index2 in range(index1 + 1, len(boxes)):
            box2 = boxes[index2]
            boxA = [box1.x1, box1.y1, box1.x2, box1.y2]
            boxB = [box2.x1, box2.y1, box2.x2, box2.y2]
            if bb_intersection_over_union(boxA, boxB) >= .05:
                boxes[index2].prob = 0
    boxes = [box for box in boxes if box.prob > 0.]
    return boxes


#draw boxes with probabilities over them
def draw_boxes(boxes, img):

    font = cv2.FONT_HERSHEY_PLAIN
    for box in boxes:

        img = cv2.rectangle(img, (box.x1, box.y1), (box.x2, box.y2), (200, 0, 0), 6)
        img = cv2.rectangle(img, (box.x1 - 3, box.y1), (box.x1 + 135, box.y1 - 35), (200, 0, 0), -6)
        img = cv2.putText(img, 'Car %{0:.3}'.format(box.prob * 100), (box.x1, box.y1 - 10), font, 1.6, (255, 255, 255), 2,
                          cv2.LINE_AA)

    return img


#get the precent of overlap of two boxes
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou




