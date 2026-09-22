"""
tools/keys.py — API key lookup.

Lookup order:
  1. tools/local_keys.py::get_local_key(service_name), if present (git-ignored, machine-local)
  2. Environment variable  {SERVICE}_API_KEY   (e.g. OPENROUTER_API_KEY)
  3. keys/key.txt  (relative to the working directory; git-ignored)
"""
import logging
import os

logger = logging.getLogger(__name__)


def get_api_key(service_name: str) -> str:
    """Return the API key for `service_name` ('openrouter', 'poe', 'openai').

    Raises:
        ConnectionError: if the key cannot be found by any method.
    """
    try:
        from tools.local_keys import get_local_key
        key = get_local_key(service_name)
        if key:
            return key
    except ImportError:
        pass

    env_var = f"{service_name.upper()}_API_KEY"
    env_key = os.getenv(env_var)
    if env_key:
        logger.info(f"Retrieved '{service_name}' key from env var {env_var}.")
        return env_key

    key_file = os.path.join(os.getcwd(), "keys", "key.txt")
    if os.path.exists(key_file):
        with open(key_file) as f:
            logger.info(f"Retrieved '{service_name}' key from {key_file}.")
            return f.read().strip()

    raise ConnectionError(
        f"No API key for '{service_name}'. Set {env_var}=<key> or put the key in keys/key.txt."
    )
