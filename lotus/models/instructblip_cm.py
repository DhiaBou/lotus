import os
from dataclasses import dataclass
from typing import List, Sequence

import torch
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

from lotus.models.cm import CM


@dataclass
class InstructBlipCaptioner(CM):
    model_name: str = "Salesforce/instructblip-flan-t5-xl"
    device: str = ("mps" if torch.backends.mps.is_available()
                   else ("cuda" if torch.cuda.is_available() else "cpu"))

    def __post_init__(self):
        torch.set_num_threads(max(1, (os.cpu_count() or 4)))
        self.processor = InstructBlipProcessor.from_pretrained(self.model_name)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype="auto", device_map=None
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def _caption_images(self, images: Sequence[Image.Image]) -> List[str]:
        if not images: return []
        prompts = ["Describe the image in detail."] * len(images)
        inputs = self.processor(images=list(images), text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model.generate(
            **inputs, do_sample=False, num_beams=4, max_new_tokens=128
        )
        return self.processor.batch_decode(out, skip_special_tokens=True)
