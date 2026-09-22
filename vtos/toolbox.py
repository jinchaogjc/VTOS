import sys, copy, types
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
from vtos.inference_cache import ImmutableEmbeddingCache
from vtos.bbox_utils import BboxNormalizer

# ── PROXY PRINT TO STDERR (Protocol Integrity) ────────────────────────
def print(*args, **kwargs):
    """Overrides print to ensure stdout remains pure JSON for worker IPC."""
    import sys
    sys.stderr.write(" ".join(map(str, args)) + "\n")

def compute_iou(boxA, boxB):
    # boxA, boxB are [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# ==========================================
# GLOBAL LAZY-LOADED RESOURCES
# ==========================================
_DINO_MODEL = None
_DINO_PROCESSOR = None
_SAM_MODEL = None
_SAM_PROCESSOR = None
_SAM2_MODEL = None
_SAM2_PROCESSOR = None
_DINO_CACHE = {}
_SAM_CACHE = {}
def _get_dino_model_and_processor(device):
    global _DINO_MODEL, _DINO_PROCESSOR
    if _DINO_MODEL is None:
        sys.stderr.write("🚀 [System] Loading Grounding DINO weights...\n")
        model_id = "IDEA-Research/grounding-dino-tiny"
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        _DINO_PROCESSOR = AutoProcessor.from_pretrained(model_id)
        _DINO_MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    return _DINO_MODEL, _DINO_PROCESSOR

def _get_sam_model_and_processor(device):
    global _SAM_MODEL, _SAM_PROCESSOR
    if _SAM_MODEL is None:
        sys.stderr.write("🚀 [System] Loading SAM weights...\n")
        model_id = "facebook/sam-vit-base"
        from transformers import SamModel, SamProcessor
        _SAM_PROCESSOR = SamProcessor.from_pretrained(model_id)
        _SAM_MODEL = SamModel.from_pretrained(model_id).to(device)
    return _SAM_MODEL, _SAM_PROCESSOR

class SAMSkill:
    def __init__(self, device=None):
        self.device = device if device else (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = None
        self.processor = None
        self.cache = None

    def segment(self, image_path, bboxes):
        if self.cache:
            res = self.cache.get(image_path, "sam_segmentation", {"bboxes": bboxes})
            if res is not None: return res
        
        if self.model is None:
            self.model, self.processor = _get_sam_model_and_processor(self.device)
        
        try:
            image = Image.open(image_path).convert("RGB") if isinstance(image_path, str) else image_path.convert("RGB")
            input_boxes = [[bboxes]]
            inputs = self.processor(image, input_boxes=input_boxes, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            masks = self.processor.image_processor.post_process_masks(
                outputs.pred_masks, inputs.original_sizes, inputs.reshaped_input_sizes
            )[0]
            if self.cache: self.cache.store(image_path, "sam_segmentation", masks, {"bboxes": bboxes})
            return masks
        except Exception as e:
            sys.stderr.write(f"❌ SAM Error: {e}\n")
            return []

def _get_sam2_model_and_processor(device, model_id="facebook/sam2-hiera-large"):
    """Lazy-load SAM 2 weights (HuggingFace transformers Sam2Model)."""
    global _SAM2_MODEL, _SAM2_PROCESSOR
    if _SAM2_MODEL is None:
        sys.stderr.write(f"🚀 [System] Loading SAM 2 weights ({model_id})...\n")
        from transformers import Sam2Model, Sam2Processor
        _SAM2_PROCESSOR = Sam2Processor.from_pretrained(model_id)
        _SAM2_MODEL = Sam2Model.from_pretrained(model_id).to(device)
        _SAM2_MODEL.eval()
    return _SAM2_MODEL, _SAM2_PROCESSOR


class Sam2Skill:
    """SAM 2 wrapper that accepts bbox prompts and returns binary masks.

    Parallels SAMSkill (SAM 1) but uses facebook/sam2-hiera-large by default.
    Returns a list of numpy uint8 masks aligned with the input bboxes — one
    mask per bbox. Empty list on failure or empty input.
    """
    def __init__(self, device=None, model_id="facebook/sam2-hiera-large"):
        self.device = device if device else (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.cache = None

    def segment(self, image_path, bboxes_xyxy_abs):
        """Run SAM 2 with absolute-pixel xyxy bboxes; return list of binary masks.

        Args:
            image_path: path or PIL.Image.
            bboxes_xyxy_abs: list of [x1, y1, x2, y2] in absolute pixel coords.

        Returns:
            List of numpy uint8 arrays, shape (H, W), aligned with input bboxes.
        """
        if not bboxes_xyxy_abs:
            return []
        # Cache lookup
        cache_key = None
        if self.cache:
            cache_key = (str(image_path), self.model_id, tuple(map(tuple, bboxes_xyxy_abs)))
            cached = self.cache.get(image_path, "sam2_segmentation", {"bboxes": bboxes_xyxy_abs})
            if cached is not None:
                return cached

        try:
            if self.model is None:
                self.model, self.processor = _get_sam2_model_and_processor(self.device, self.model_id)

            image = Image.open(image_path).convert("RGB") if isinstance(image_path, str) else image_path.convert("RGB")
            # Sam2Processor expects input_boxes shaped [batch, num_objects, 4]
            input_boxes = [[list(b) for b in bboxes_xyxy_abs]]
            inputs = self.processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, multimask_output=False)
            masks = self.processor.post_process_masks(
                outputs.pred_masks.cpu(),
                original_sizes=inputs["original_sizes"].cpu()
                                if "original_sizes" in inputs else [image.size[::-1]],
            )[0]
            # `masks` shape: (num_boxes, num_masks_per_box=1, H, W) or (num_boxes, H, W)
            out_masks = []
            for m in masks:
                arr = m.numpy() if hasattr(m, "numpy") else np.asarray(m)
                if arr.ndim == 3:  # squeeze multi-mask dim
                    arr = arr[0]
                out_masks.append((arr > 0).astype(np.uint8))
            if self.cache:
                self.cache.store(image_path, "sam2_segmentation", out_masks, {"bboxes": bboxes_xyxy_abs})
            return out_masks
        except Exception as e:
            sys.stderr.write(f"❌ SAM 2 Error: {e}\n")
            return []

    def segment_points(self, image_path, points_xy_abs):
        """Run SAM 2 with absolute-pixel point prompts; return list of binary masks.

        Each point is treated as one positive (foreground) prompt for one object.

        Args:
            image_path: path or PIL.Image.
            points_xy_abs: list of [x, y] in absolute pixel coords.

        Returns:
            List of numpy uint8 masks, one per input point.
        """
        if not points_xy_abs:
            return []
        try:
            if self.model is None:
                self.model, self.processor = _get_sam2_model_and_processor(self.device, self.model_id)
            image = Image.open(image_path).convert("RGB") if isinstance(image_path, str) else image_path.convert("RGB")
            # Sam2Processor input_points: [batch, num_objects, num_points_per_object, 2];
            # one point per object → one object per input point.
            input_points = [[[[float(p[0]), float(p[1])]]] for p in points_xy_abs]
            input_points = [[obj[0] for obj in input_points]]   # [batch][num_objects][1][2]
            input_labels = [[[1] for _ in points_xy_abs]]       # all foreground
            inputs = self.processor(images=image, input_points=input_points,
                                    input_labels=input_labels, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, multimask_output=False)
            masks = self.processor.post_process_masks(
                outputs.pred_masks.cpu(),
                original_sizes=inputs["original_sizes"].cpu()
                                if "original_sizes" in inputs else [image.size[::-1]],
            )[0]
            out_masks = []
            for m in masks:
                arr = m.numpy() if hasattr(m, "numpy") else np.asarray(m)
                if arr.ndim == 3:
                    arr = arr[0]
                out_masks.append((arr > 0).astype(np.uint8))
            return out_masks
        except Exception as e:
            sys.stderr.write(f"❌ SAM 2 point-prompt Error: {e}\n")
            return []


class GroundingDINOSkill:
    def __init__(self, device=None):
        self.device = device if device else (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = None
        self.processor = None
        self.cache = None

    def detect(self, image_path, text_query, box_threshold=0.3):
        if self.cache:
            res = self.cache.get(image_path, text_query, {"threshold": box_threshold})
            if res is not None: return res
        
        if self.model is None:
            self.model, self.processor = _get_dino_model_and_processor(self.device)
            
        try:
            image = Image.open(image_path).convert("RGB") if isinstance(image_path, str) else image_path.convert("RGB")
            query = text_query.strip() + "." if not text_query.strip().endswith(".") else text_query.strip()
            inputs = self.processor(images=image, text=query, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=box_threshold, target_sizes=[image.size[::-1]]
            )[0]
            out = (results["boxes"].cpu().numpy().tolist(), results["scores"].cpu().numpy().tolist(), results.get("labels", []))
            if self.cache: self.cache.store(image_path, text_query, out, {"threshold": box_threshold})
            return out
        except Exception as e:
            sys.stderr.write(f"❌ DINO Error: {e}\n")
            return [], [], []

class ToolboxWrapper:
    def __init__(self, skill_field_ref):
        self.field = skill_field_ref
        self.cache = ImmutableEmbeddingCache()
        self._dynamic_tools = {}
        
        self.dino_skill = GroundingDINOSkill()
        self.dino_skill.cache = self.cache
        self.sam_skill = SAMSkill()
        self.sam_skill.cache = self.cache
        self.latest_bboxes = []
        self._load_skills()

    def _load_skills(self):
        import importlib, pkgutil
        import vtos.skills as skills
        for loader, module_name, is_pkg in pkgutil.iter_modules(skills.__path__):
            full_name = f"vtos.skills.{module_name}"
            try:
                module = importlib.import_module(full_name)
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if callable(attr) and not attr_name.startswith("_"):
                        setattr(self, attr_name, types.MethodType(attr, self))
                        self._dynamic_tools[attr_name] = {"doc": attr.__doc__ or ""}
            except Exception as e:
                sys.stderr.write(f"❌ Skill Load Error ({module_name}): {e}\n")

    def _force_normalize(self, bboxes, img_w, img_h):
        if not bboxes: return []
        sanitized = []
        for box in bboxes:
            if len(box) != 4: continue
            x1, y1, x2, y2 = box
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            if max(xmin, ymin, xmax, ymax) < 0.05:
                xmin, xmax, ymin, ymax = xmin*img_w, xmax*img_w, ymin*img_h, ymax*img_h
            elif max(xmin, ymin, xmax, ymax) > 1.5:
                xmin, xmax, ymin, ymax = xmin/img_w, xmax/img_w, ymin/img_h, ymax/img_h
            sanitized.append([max(0.0, min(1.0, xmin)), max(0.0, min(1.0, ymin)), 
                             max(0.0, min(1.0, xmax)), max(0.0, min(1.0, ymax))])
        return sanitized

    def grounding_dino_detect(self, image, text_query, box_threshold=0.3, **kwargs):
        query = text_query or kwargs.get('query')
        if not query: raise ValueError("Missing text_query")
        
        img_key = image if isinstance(image, str) else str(id(image))
        cache_key = f"{img_key}_{query}_{box_threshold}_v5"
        global _DINO_CACHE
        if cache_key in _DINO_CACHE: return copy.deepcopy(_DINO_CACHE[cache_key])
        
        bboxes, scores, labels = self.dino_skill.detect(image, query, box_threshold)
        bboxes = BboxNormalizer.purge_phantom_boxes(bboxes)
        self.latest_bboxes = bboxes
        
        if isinstance(image, str):
            from PIL import Image
            with Image.open(image) as im: w, h = im.size
        else: w, h = image.size
        
        final = self._force_normalize(bboxes, w, h)
        _DINO_CACHE[cache_key] = copy.deepcopy(final)
        return final

    def sam_refine_bboxes(self, image, bboxes):
        if not bboxes: return []
        img_key = image if isinstance(image, str) else str(id(image))
        cache_key = f"{img_key}_{str(bboxes)}_v5"
        global _SAM_CACHE
        if cache_key in _SAM_CACHE: return copy.deepcopy(_SAM_CACHE[cache_key])
        
        if isinstance(image, str):
            from PIL import Image
            with Image.open(image) as im: w, h = im.size
        else: w, h = image.size
        
        norm_boxes = self._force_normalize(bboxes, w, h)
        abs_boxes = [[x1*w, y1*h, x2*w, y2*h] for x1,y1,x2,y2 in norm_boxes]
        masks = self.sam_skill.segment(image, abs_boxes)
        abs_boxes = BboxNormalizer.purge_phantom_boxes(abs_boxes)
        final = self._force_normalize(abs_boxes, w, h)
        _SAM_CACHE[cache_key] = copy.deepcopy(final)
        return final

    def compute_iou(self, boxA, boxB):
        """Compute IoU between two boxes [x1, y1, x2, y2] in normalized coords."""
        xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return inter / float(areaA + areaB - inter + 1e-6)

    def __getattr__(self, name):
        if name in self._dynamic_tools: return getattr(self, name)
        raise AttributeError(f"No tool named {name}")
