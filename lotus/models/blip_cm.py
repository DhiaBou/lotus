from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from lotus.models.cm import CM


@dataclass
class BlipCaptioner(CM):
    model_name: str = "Salesforce/blip-image-captioning-large"
    device: str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        torch.set_num_threads(max(1, (os.cpu_count() or 4)))
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def _caption_images(self, images: Sequence[Image.Image]) -> List[str]:
        if not images:
            return []
        inputs = self.processor(images=list(images), return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model.generate(**inputs, 
                                  do_sample=False, 
                                  early_stopping=False,
                                  length_penalty=0.9,
                                    num_beams=3,
                                    no_repeat_ngram_size=3,
                                    repetition_penalty=1.07,
                                    max_new_tokens=70,
                                    renormalize_logits=True,
                                  )
        return self.processor.batch_decode(out, skip_special_tokens=True)
