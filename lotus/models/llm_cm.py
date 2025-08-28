from dataclasses import dataclass
from typing import Callable, List, Sequence, Optional

import pandas as pd
from PIL import Image

import lotus
from lotus.dtype_extensions import ImageArray
from lotus.models.cm import CM
from lotus.sem_ops.postprocessors import map_postprocess
from lotus.sem_ops.sem_map import sem_map
from lotus.templates import task_instructions
from lotus.types import ReasoningStrategy


@dataclass
class LLMCaptioner(CM):
    """
    Caption images by calling your LLM through `sem_map`.

    - Uses lotus.settings.lm by default (or pass a model explicitly).
    - Converts a batch of PIL Images -> `multimodal_data` via df2multimodal_info.
    """
    model: Optional[lotus.models.LM] = None
    user_instruction: str = "Describe the image in detail without guessing."
    strategy: Optional[ReasoningStrategy] = None
    postprocessor: Callable[[list[str], lotus.models.LM, bool], object] = map_postprocess
    safe_mode: bool = False
    progress_bar_desc: str = "Captioning"

    def _caption_images(self, images: Sequence[Image.Image]) -> List[str]:
        if not images:
            return []

        df = pd.DataFrame({"image": ImageArray(images)})

        multimodal_data = task_instructions.df2multimodal_info(df, ["image"])

        output = sem_map(
            docs=multimodal_data,
            model=lotus.settings.lm,
            user_instruction=self.user_instruction,
            postprocessor=self.postprocessor,
            strategy=self.strategy,
            safe_mode=self.safe_mode,
            progress_bar_desc=self.progress_bar_desc,
        )

        return list(output.outputs)
