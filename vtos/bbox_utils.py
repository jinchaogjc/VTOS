
class BboxNormalizer:
    """
    Handles deterministic transformations between absolute pixel space
    and normalized coordinate space for bounding box evaluations.
    """

    @staticmethod
    def purge_phantom_boxes(raw_bboxes):
        """
        Eliminates LLM hallucinated inverted/negative-area bounding boxes.
        Assumes [xmin, ymin, xmax, ymax]
        """
        if not raw_bboxes:
            return []
            
        valid_boxes = []
        for box in raw_bboxes:
            if len(box) != 4:
                continue
            xmin, ymin, xmax, ymax = box
            
            # Kill inverted boxes
            if xmin >= xmax or ymin >= ymax:
                continue
                
            # Ensure physical area
            if (xmax - xmin) > 1e-4 and (ymax - ymin) > 1e-4:
                valid_boxes.append(box)
                
        return valid_boxes

