# captioner.py
from dataclasses import dataclass
from typing import List, Sequence
from PIL import Image
import torch

# Swap to any caption model you like (BLIP, GIT, Florence, etc.)
from transformers import BlipProcessor, BlipForConditionalGeneration

@dataclass
class Captioner:
    model_name: str = "Salesforce/blip-image-captioning-large"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, images: Sequence[Image.Image]) -> List[str]:
        if not images:
            return []
        inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        return self.processor.batch_decode(out, skip_special_tokens=True)
