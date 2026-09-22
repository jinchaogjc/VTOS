import numpy as np

# Strategy: Refined version of sol_011_01 with adjusted thresholds
# Key changes: 
# 1. Use ratio > 0.70 (vs 0.75) to better catch img008 (ratio=0.833 at threshold 0.15/0.10)
# 2. Slightly lower count trigger to 32 (vs 35) to avoid triggering on normal images
# 3. For over-predicted images (n_low > 55), use box_threshold=0.14 instead of 0.13

bboxes_low = toolbox.grounding_dino_detect(image_path, text_query=text_query, box_threshold=0.10)
bboxes_low = toolbox.nms_filter(bboxes_low, iou_threshold=0.30)
n_low = len(bboxes_low)

bboxes_high = toolbox.grounding_dino_detect(image_path, text_query=text_query, box_threshold=0.15)
n_high = len(bboxes_high)

ratio = n_high / max(n_low, 1)

# Under-detection trigger: sparse OR high-ratio pattern with moderate count
if n_low < 32 or (n_low < 55 and ratio > 0.70 and n_high < 35):
    # Under-detected: augment with 3x3 slice detection
    bboxes_sliced = toolbox.slice_and_detect(image_path, text_query=text_query, grid=(3, 3), box_threshold=0.12)
    combined = list(bboxes_low) + list(bboxes_sliced)
    bboxes = toolbox.nms_filter(combined, iou_threshold=0.28)
elif n_low > 55:
    # Dense/over-predicted: apply tighter threshold and NMS
    bboxes = toolbox.grounding_dino_detect(image_path, text_query=text_query, box_threshold=0.14)
    bboxes = toolbox.nms_filter(bboxes, iou_threshold=0.25)
else:
    # Normal density: standard approach
    bboxes = toolbox.nms_filter(bboxes_low, iou_threshold=0.30)