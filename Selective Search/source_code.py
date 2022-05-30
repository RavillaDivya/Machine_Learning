import cv2
import os
import argparse
import numpy as np
import xml.etree.ElementTree as ET

def selective_search(img, strategy):
    """
    @brief Selective search with different strategies
    @param img The input image
    @param strategy The strategy selected ['color', 'all']
    @retval bboxes Bounding boxes
    """
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    gs = cv2.ximgproc.segmentation.createGraphSegmentation()

    #Setting K and Sigma value for the Selective Search
    gs.setK(200)
    gs.setSigma(0.8)

    #Setting strategy for the Selective Search
    if strategy == 'color':
       strategy = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()

    else:
       s1 = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
       s2 = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill()
       s3 = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize()
       s4 = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture()
		
       strategy = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple(s1,s2,s3,s4)

    ss.addStrategy(strategy)	
    ss.addGraphSegmentation(gs)

    #Feeding RGB image into the SS object
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ss.addImage(img)
	
    bboxes = ss.process()
    xyxy_bboxes = []

    for box in bboxes:
        x, y, w, h = box
        xyxy_bboxes.append([x, y, x+w, y + h])

    return xyxy_bboxes

def parse_annotation(anno_path):
    """
    @brief Parse annotation files for ground truth bounding boxes
    @param anno_path Path to the file
    """
    tree = ET.parse(anno_path)
    root = tree.getroot()
    gt_bboxes = []
    for child in root:
        if child.tag == 'object':
            for grandchild in child:
                if grandchild.tag == "bndbox":
                    x0 = int(grandchild.find('xmin').text)
                    x1 = int(grandchild.find('xmax').text)
                    y0 = int(grandchild.find('ymin').text)
                    y1 = int(grandchild.find('ymax').text)
                    gt_bboxes.append([x0, y0, x1, y1])
    return gt_bboxes

def bb_intersection_over_union(boxA, boxB):
    """
    @brief compute the intersaction over union (IoU) of two given bounding boxes
    @param boxA numpy array (x_min, y_min, x_max, y_max)
    @param boxB numpy array (x_min, y_min, x_max, y_max)
    """


    #Calculating the area of the two boxes to find the union
    area1 = (boxA[2] - boxA[0])*(boxA[3] - boxA[1])
    area2 = (boxB[2] - boxB[0])*(boxB[3] - boxB[1])

    #Locating the boundaries to find intersection
    xleft = max(boxA[0],boxB[0])
    xright = min(boxA[2],boxB[2])
    ybottom = min(boxA[3],boxB[3])
    ytop = max(boxA[1],boxB[1])

    #Condition to check if the two boxes are mutually exclusive(intersection=0)
    if xright < xleft or ybottom < ytop:
       return 0.0

    intersection = (xright - xleft)*(ybottom - ytop)

    #Calculating iou using (intersection/union) formula
    iou = intersection/ float(area1 + area2 - intersection)
 
    return abs(iou)

def visualize(img, boxes, color):
    """
    @breif Visualize boxes
    @param img The target image
    @param boxes The box list
    @param color The color
    """
    for box in boxes:
        #print(box)
        img = cv2.rectangle(img, box[:2], box[2:], color)

    return img


def main():
    parser = argparse.ArgumentParser()

    # Change default to 'all' when we have to use all strategies
    parser.add_argument('--strategy', type=str, default='color')

    args =parser.parse_args()
    img_dir = './HW2_Data/JPEGImages'
    anno_dir = './HW2_Data/Annotations'
    thres = .5

    img_no = 0

    img_list = os.listdir(img_dir)
    num_hit = 0
    num_gt = 0

    for img_path in img_list:
        """
        Load the image file here through cv2.imread
        """
        img_id = img_path[:-4]
        img_name = os.path.join(img_dir, img_path)

        img = cv2.imread(img_name)

        #Used for naming the output images
        img_no += 1

        proposals = selective_search(img, args.strategy)
        gt_bboxes = parse_annotation(os.path.join(anno_dir, img_id + ".xml"))
        iou_bboxes = []  # proposals with IoU greater than 0.5

	    #for each ground truth boxes we look for proposal boxes that have IoU >= 0.5
        for boxA in gt_bboxes:
           max_iou = 0.0
           max_box = np.array([])
           for boxB in proposals:
              iou = bb_intersection_over_union(boxA,boxB)
              if iou >= thres and iou > max_iou:
                 max_box = boxB
                 max_iou = iou
           if len(max_box):
              #print(max_box)
              #append only if the max_box is not empty
              iou_bboxes.append(max_box)

        #recall is calculated as (number of IoUboxes detected with the threshold)/(number of ground truth boxes)
        recall = len(iou_bboxes)/len(gt_bboxes)
        print('recall',recall)
        
        vis_img = img.copy()
        vis_img = visualize(vis_img, gt_bboxes, (255, 0, 0))
        vis_img = visualize(vis_img, iou_bboxes, (0, 0, 255))

        proposals_img = img.copy()
        proposals_img = visualize(proposals_img, gt_bboxes, (255, 0, 0))
        proposals_img = visualize(proposals_img, proposals, (0, 0, 255))

        s_name = 'IOUThreshold' + str(img_no) + '.jpg'
        p_name = 'Proposals' + str(img_no) + '.jpg'


        #Saving the images
        cv2.imwrite(s_name, vis_img)
        cv2.imwrite(p_name, proposals_img)
        


if __name__ == "__main__":
    main()




