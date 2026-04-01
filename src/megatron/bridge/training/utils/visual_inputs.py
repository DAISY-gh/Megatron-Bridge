# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import torch


@dataclass
class Qwen2_5_VLVisualInputs:
    """Container for Qwen2/Qwen2.5-VL visual modality tensors.

    Fields mirror the processor outputs for Qwen2/Qwen2.5-VL. Shapes may be
    normalized for model consumption via normalized_for_model().
    """

    pixel_values: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None
    pixel_values_videos: Optional[torch.Tensor] = None
    video_grid_thw: Optional[torch.Tensor] = None

    def as_model_kwargs(self) -> dict[str, torch.Tensor]:
        """Return a mapping of non-None fields suitable for model forward kwargs."""
        result: dict[str, torch.Tensor] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                result[f.name] = value
        return result

    def normalized_for_model(self) -> dict[str, torch.Tensor]:
        """Return non-None fields with shapes normalized for model expectations.

        Flattens batch dimensions for tensors that have them:
        - pixel_values / pixel_values_videos: [B, N, C, H, W] -> [B*N, C, H, W]
        - image_grid_thw / video_grid_thw: [B, N, 3] -> [B*N, 3]
        """
        kwargs = self.as_model_kwargs()

        for pv_key in ("pixel_values", "pixel_values_videos"):
            pv = kwargs.get(pv_key)
            if isinstance(pv, torch.Tensor) and pv.dim() == 5:
                b, n, c, h, w = pv.shape
                kwargs[pv_key] = pv.view(b * n, c, h, w)

        for thw_key in ("image_grid_thw", "video_grid_thw"):
            thw = kwargs.get(thw_key)
            if isinstance(thw, torch.Tensor) and thw.dim() == 3:
                kwargs[thw_key] = thw.view(-1, thw.size(-1))

        return kwargs
