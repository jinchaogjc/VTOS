# sol_004_01 — train score=0.4002

# Contextual query + loose NMS=0.55 to push co-located lesion preservation further
query = f"{target} on {plant}"
bboxes = detect(query, threshold=0.15)
bboxes = filter_by_area(bboxes, min_norm_area=0.0002, max_norm_area=0.80)
bboxes = nms(bboxes, iou_thresh=0.55)
final_bboxes = bboxes
polygons = segment(final_bboxes)
polygons = filter_polygons_by_area(polygons, min_norm_area=0.0001, max_norm_area=0.85)
polygons = clean_mask(polygons)
final_polygons = polygons
