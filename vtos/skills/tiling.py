import os, uuid, tempfile
from PIL import Image

def slice_and_detect(toolbox, image_path: str, text_query: str = None, grid: tuple = (2, 2), box_threshold: float = 0.25, iou_threshold: float = 0.2, **kwargs):
    """
    Atomic Skill: Grid-slice the image to detect extraordinarily small or dense objects.
    """
    actual_query = text_query or kwargs.get('query')
    if not actual_query:
        raise ValueError("slice_and_detect missing required argument: 'text_query' or 'query'")
    
    import sys
    sys.stderr.write(f"🔪 [Atomic Skill] Slice & Detect Triggered: {grid[0]}x{grid[1]} grid for '{actual_query}'\n")
    
    original_image = Image.open(image_path).convert("RGB") if isinstance(image_path, str) else image_path.convert("RGB")
    img_width, img_height = original_image.size
    
    rows, cols = grid
    slice_width = img_width // cols
    slice_height = img_height // rows
    
    global_abs_bboxes = []
    global_scores = []
    temp_dir = tempfile.gettempdir()
    session_id = str(uuid.uuid4())[:8]
    
    for row in range(rows):
        for col in range(cols):
            left = col * slice_width
            upper = row * slice_height
            right = left + slice_width if col < cols - 1 else img_width
            lower = upper + slice_height if row < rows - 1 else img_height
            
            slice_img = original_image.crop((left, upper, right, lower))
            temp_path = os.path.join(temp_dir, f"slice_{session_id}_{row}_{col}.jpg")
            slice_img.save(temp_path)
            
            try:
                local_tuple = toolbox.dino_skill.detect(temp_path, actual_query, box_threshold)
                if local_tuple and len(local_tuple) >= 2:
                    for i, box in enumerate(local_tuple[0]):
                        is_norm = max(box) <= 1.01
                        sw, sh = right - left, lower - upper
                        if is_norm:
                            gx = (box[0] * sw) + left
                            gy = (box[1] * sh) + upper
                            gx2 = (box[2] * sw) + left
                            gy2 = (box[3] * sh) + upper
                        else:
                            gx, gy, gx2, gy2 = box[0]+left, box[1]+upper, box[2]+left, box[3]+upper
                        global_abs_bboxes.append([gx, gy, gx2, gy2])
                        global_scores.append(local_tuple[1][i])
            except Exception as e:
                print(f"⚠️ Slice detection failed: {e}")
            finally:
                if os.path.exists(temp_path): os.remove(temp_path)
                    
    # Use toolbox NMS for cleanup
    return toolbox.nms_filter(global_abs_bboxes, iou_threshold=iou_threshold, image=original_image)
