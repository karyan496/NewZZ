import logging
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

# all-MiniLM-L6-v2 → 384-dim, fast, free, no API cost
_model = None


def get_model() -> SentenceTransformer:
    """Lazy-load the model so it's only downloaded once."""
    global _model
    if _model is None:
        logger.info("Loading sentence-transformer model (first run may download ~90MB)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Model loaded.")
    return _model


def get_embedding(text: str) -> np.ndarray:
    """Return a 384-dim float32 numpy vector for the given text."""
    model = get_model()
    return model.encode(text, convert_to_numpy=True).astype("float32")


def get_profile_text(user_profile: dict) -> str:
    """
    Flatten a user profile dict into a single string for embedding.
    Mirrors the structure in user_profile.py.
    """
    interests = " ".join(user_profile.get("interests", []))
    preferences = " ".join(
        k for k, v in user_profile.get("preferences", {}).items() if v is True
    )
    return (
        f"{user_profile.get('title', '')} "
        f"{user_profile.get('background', '')} "
        f"{interests} "
        f"{preferences} "
        f"{user_profile.get('expertise_level', '')}"
    )