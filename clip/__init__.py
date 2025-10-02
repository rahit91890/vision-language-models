"""CLIP (Contrastive Language-Image Pre-training) module.

This module provides integration with OpenAI's CLIP model for
vision-language understanding tasks.
"""

from .model import CLIPModel
from .utils import preprocess_image, encode_text

__all__ = ['CLIPModel', 'preprocess_image', 'encode_text']
