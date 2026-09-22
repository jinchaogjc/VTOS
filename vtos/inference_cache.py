import hashlib
import json
from typing import Any, Dict, Optional

class ImmutableEmbeddingCache:
    """
    Implements Section 2.3: Cached Foundation Model Inference Layer.
    Stores and retrieves exact Vision Foundation Model inferences (DINOv2 embeddings, SAM masks)
    in O(1) time based on deterministic Hash inputs to decouple execution latency from search depth.
    """
    
    def __init__(self, disk_cache_dir: Optional[str] = None):
        # Default to <repo>/assets/cache
        if disk_cache_dir is None:
            import os
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            disk_cache_dir = os.path.join(root, "assets/cache")
            
        self._cache: Dict[str, Any] = {}
        self.disk_cache_dir = disk_cache_dir
        
        if self.disk_cache_dir:
            import os
            os.makedirs(self.disk_cache_dir, exist_ok=True)
        
    def _compute_hash(self, image_path: str, query: str = "", kwargs: Optional[Dict] = None) -> str:
        """
        Computes a deterministic MD5 hash string based on the invariant image path, 
        the specific prompt/query, and hyper-parameters like thresholds.
        """
        # We must serialize the kwargs strictly to ensure ordering doesn't break hashes
        kwargs_str = ""
        if kwargs:
            kwargs_str = json.dumps(kwargs, sort_keys=True)
            
        # If it's a PIL Image directly in memory we can't hash the path easily.
        # We assume image_path is a file path string 99% of the time in this framework.
        img_id = str(image_path)
        
        raw_string = f"{img_id}_{query}_{kwargs_str}"
        return hashlib.md5(raw_string.encode('utf-8')).hexdigest()
        
    def get(self, image_path: str, query: str = "", kwargs: Optional[Dict] = None) -> Optional[Any]:
        """
        Retrieves cached Foundation Model output if it exists. Returns None if cache miss.
        """
        h = self._compute_hash(image_path, query, kwargs)
        
        # 1. Check in-memory cache
        if h in self._cache:
            return self._cache[h]
            
        # 2. Check disk cache
        if self.disk_cache_dir:
            import os, pickle
            from filelock import FileLock
            disk_path = os.path.join(self.disk_cache_dir, f"{h}.pkl")
            lock_path = f"{disk_path}.lock"
            
            if os.path.exists(disk_path):
                try:
                    with FileLock(lock_path, timeout=10):
                        with open(disk_path, 'rb') as f:
                            data = pickle.load(f)
                    self._cache[h] = data # Load into RAM for next time
                    return data
                except Exception as e:
                    print(f"⚠️ Failed to load disk cache for {h}: {e}")
                    
        return None
        
    def store(self, image_path: str, query: str, output: Any, kwargs: Optional[Dict] = None):
        """
        Stores Foundation Model output persistently in the memory map and disk.
        Uses FileLock and Atomic Rename to prevent corruption during parallel multiprocessing.
        """
        h = self._compute_hash(image_path, query, kwargs)
        self._cache[h] = output
        
        if self.disk_cache_dir:
            import os, pickle, tempfile
            from filelock import FileLock
            disk_path = os.path.join(self.disk_cache_dir, f"{h}.pkl")
            lock_path = f"{disk_path}.lock"
            
            try:
                with FileLock(lock_path, timeout=10):
                    # Write to a temporary file first for atomicity
                    fd, temp_path = tempfile.mkstemp(dir=self.disk_cache_dir, prefix=f"{h}_", suffix=".tmp")
                    with os.fdopen(fd, 'wb') as f:
                        pickle.dump(output, f)
                    
                    # Atomic replace prevents corrupted reads from other processes
                    os.replace(temp_path, disk_path)
            except Exception as e:
                print(f"⚠️ Failed to save disk cache for {h}: {e}")
                # Cleanup temp file if something failed
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
        
    def clear(self):
        """Clears the evaluation cache. Useful when switching to entirely new tasklets."""
        self._cache.clear()
