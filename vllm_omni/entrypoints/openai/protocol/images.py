# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OpenAI-compatible protocol definitions for image generation.

This module provides Pydantic models that follow the OpenAI DALL-E API specification
for text-to-image generation, with vllm-omni specific extensions.
"""

import base64
import io
import uuid
import zipfile
from enum import Enum
from http import HTTPStatus
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator


class ResponseFormat(str, Enum):
    """Image response format"""

    B64_JSON = "b64_json"
    URL = "url"  # Not implemented in PoC
    FILE = "file"


class ImageGenerationRequest(BaseModel):
    """
    OpenAI DALL-E compatible image generation request.

    Follows the OpenAI Images API specification with vllm-omni extensions
    for advanced diffusion parameters.
    """

    # Required fields
    prompt: str = Field(..., description="Text description of the desired image(s)")

    # OpenAI standard fields
    model: str | None = Field(
        default=None,
        description="Model to use (optional, uses server's configured model if omitted)",
    )
    n: int = Field(default=1, ge=1, le=10, description="Number of images to generate")
    size: str | None = Field(
        default=None,
        description="Image dimensions in WIDTHxHEIGHT format (e.g., '1024x1024', uses model defaults if omitted)",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.B64_JSON, description="Format of the returned image")
    user: str | None = Field(default=None, description="User identifier for tracking")

    @field_validator("size")
    @classmethod
    def validate_size(cls, v):
        """Validate size parameter.

        Accepts any string in 'WIDTHxHEIGHT' format (e.g., '1024x1024', '512x768').
        No restrictions on specific dimensions - models can handle arbitrary sizes.
        """
        if v is None:
            return None
        # Validate string format
        if not isinstance(v, str) or "x" not in v:
            raise ValueError("size must be in format 'WIDTHxHEIGHT' (e.g., '1024x1024')")
        return v

    @field_validator("response_format")
    @classmethod
    def validate_response_format(cls, v):
        """Validate response format - only b64_json is supported."""
        if v is not None and v != ResponseFormat.B64_JSON and v != ResponseFormat.FILE:
            raise ValueError(f"Only 'b64_json' response format is supported, got: {v}")
        return v

    # vllm-omni extensions for diffusion control
    negative_prompt: str | None = Field(default=None, description="Text describing what to avoid in the image")
    num_inference_steps: int | None = Field(
        default=None,
        ge=1,
        le=200,
        description="Number of diffusion sampling steps (uses model defaults if not specified)",
    )
    guidance_scale: float | None = Field(
        default=None,
        ge=0.0,
        le=20.0,
        description="Classifier-free guidance scale (uses model defaults if not specified)",
    )
    true_cfg_scale: float | None = Field(
        default=None,
        ge=0.0,
        le=20.0,
        description="True CFG scale (model-specific parameter, may be ignored if not supported)",
    )
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    generator_device: str | None = Field(
        default=None,
        description="Device for the seeded torch.Generator (e.g. 'cpu', 'cuda'). Defaults to the runner's device.",
    )

    # vllm-omni extension for per-request LoRA.
    # This mirrors the `extra_body.lora` convention in /v1/chat/completions.
    lora: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional LoRA adapter for this request. Expected shape: "
            "{name/path/scale/int_id}. Field names are flexible "
            "(e.g. name|lora_name|adapter, path|lora_path|local_path, "
            "scale|lora_scale, int_id|lora_int_id)."
        ),
    )

    # VAE memory optimizations (set at model init, included for completeness)
    vae_use_slicing: bool | None = Field(default=False, description="Enable VAE slicing")
    vae_use_tiling: bool | None = Field(default=False, description="Enable VAE tiling")


class ImageData(BaseModel):
    """Single generated image data"""

    b64_json: str | None = Field(default=None, description="Base64-encoded PNG image")
    url: str | None = Field(default=None, description="Image URL (not implemented)")
    revised_prompt: str | None = Field(default=None, description="Revised prompt (OpenAI compatibility, always null)")


class ImageGenerationResponse(BaseModel):
    """
    OpenAI DALL-E compatible image generation response.

    Returns generated images with metadata.
    """

    created: int = Field(..., description="Unix timestamp of when the generation completed")
    data: list[ImageData] = Field(..., description="Array of generated images")
    output_format: str = Field(None, description="The output format of the image generation")
    size: str = Field(None, description="The size of the image generated")

    def stream_response(self) -> StreamingResponse:
        if not self.data or not self.data[0].b64_json:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                detail="No image data available for file response.",
            )
        if len(self.data) == 1:
            image_bytes = base64.b64decode(self.data[0].b64_json)
            filename = f"image_{uuid.uuid4().hex[:8]}.png"
            return StreamingResponse(
                io.BytesIO(image_bytes),
                media_type="image/png",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"',
                    "Content-Length": str(len(image_bytes)),
                },
            )
        else:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for idx, item in enumerate(response.data):
                    if item.b64_json:
                        zf.writestr(f"image_{idx}.png", base64.b64decode(item.b64_json))
            zip_bytes = zip_buffer.getvalue()
            filename = f"images_{uuid.uuid4().hex[:8]}.zip"
            return StreamingResponse(
                io.BytesIO(zip_bytes),
                media_type="application/zip",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"',
                    "Content-Length": str(len(zip_bytes)),
                },
            )
