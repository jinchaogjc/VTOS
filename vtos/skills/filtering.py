import numpy as np

def nms_filter(toolbox, bboxes: list, iou_threshold: float = 0.45, image=None):
    """
    Atomic Skill: Explicit Non-Maximum Suppression (NMS) for overlapping boxes.
    """
    if not bboxes or len(bboxes) == 0:
        return []
        
    boxes_array = np.array(bboxes, dtype=np.float32)
    x1, y1, x2, y2 = boxes_array[:, 0], boxes_array[:, 1], boxes_array[:, 2], boxes_array[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.arange(len(boxes_array))
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
        
    filtered_boxes = boxes_array[keep].tolist()
    
    if image is not None:
        if isinstance(image, str):
            from PIL import Image
            img_obj = Image.open(image).convert("RGB")
        else:
            img_obj = image.convert("RGB")
        img_w, img_h = img_obj.size
        return toolbox._force_normalize(filtered_boxes, img_w, img_h)
        
    return filtered_boxes

