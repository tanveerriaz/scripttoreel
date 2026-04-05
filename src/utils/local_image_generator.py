"""
Local AI image generation via Hugging Face diffusers + Apple MPS.

Uses Stable Diffusion XL (SDXL) on Mac M-series via Metal Performance Shaders.
Pipeline is lazy-loaded and class-level cached — only loaded once per process.

Install (one-time):
    pip install torch diffusers transformers accelerate

First run downloads ~6GB SDXL model to ~/.cache/huggingface (cached forever after).
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from src.utils.json_schemas import (
    Asset,
    AssetRole,
    AssetSource,
    AssetType,
    LicensingInfo,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Genre → cinematic style modifiers
# ---------------------------------------------------------------------------

_GENRE_MODIFIERS: dict[str, str] = {
    "thriller":    "cinematic still, dramatic lighting, film noir, 8K, photorealistic, high contrast",
    "horror":      "dark atmospheric, moonlit, eerie mist, horror movie still, photorealistic",
    "action":      "dynamic composition, motion blur, cinematic action still, golden hour, high contrast",
    "spy":         "espionage thriller, covert ops, night scene, cinematic, photorealistic",
    "heist":       "cinematic heist still, tense atmosphere, urban night, photorealistic",
    "documentary": "documentary photography, natural light, journalistic, candid, real location",
    "educational": "clean, well-lit, professional setting, informative, photorealistic",
    "birthday":    "vibrant celebration, bokeh background, warm golden lighting, professional photography",
    "celebration": "joyful, colourful, festive atmosphere, warm lighting, professional photography",
    "wedding":     "elegant, soft bokeh, golden hour, romantic, professional wedding photography",
    "science":     "laboratory, high-tech equipment, clean scientific aesthetic, photorealistic",
    "nature":      "National Geographic style, golden hour, dramatic landscape, photorealistic",
    "travel":      "travel photography, iconic location, blue hour, sharp detail, cinematic",
    "business":    "corporate professional, modern office, business magazine style, photorealistic",
    "food":        "food photography, shallow depth of field, moody restaurant lighting, appetizing",
    "history":     "historical, archival aesthetic, sepia tones, dramatic lighting, photorealistic",
}

_DEFAULT_MODIFIER = "cinematic photography, professional lighting, 8K, photorealistic, high detail"

_NEGATIVE_PROMPT = (
    "cartoon, illustration, 3d render, anime, painting, drawing, sketch, "
    "watermark, text, logo, nude, nsfw, blurry, low quality, low resolution, "
    "out of focus, grainy, distorted, deformed, ugly, bad anatomy"
)


def build_image_prompt(query: str, topic: str, tone: str = "") -> str:
    """Build an enhanced SDXL prompt from a stock search query + topic context.

    Detects genre/tone keywords and appends matching cinematic style modifiers.
    Example:
        query="night chase rooftop Karachi", topic="Pakistani action thriller"
        → "night chase rooftop Karachi, cinematic still, dramatic lighting, film noir, 8K, photorealistic"
    """
    combined = f"{topic} {tone}".lower()
    modifier = _DEFAULT_MODIFIER
    for keyword, mod in _GENRE_MODIFIERS.items():
        if keyword in combined:
            modifier = mod
            break
    return f"{query}, {modifier}"


# ---------------------------------------------------------------------------
# LocalSDXLClient
# ---------------------------------------------------------------------------

class LocalSDXLClient:
    """Generates images via Stable Diffusion XL on Apple MPS (M-series Mac).

    The diffusers pipeline is lazy-loaded on first call and cached at the
    class level so it is only loaded once per process — ~5s load time after
    the first run (model is cached by HuggingFace in ~/.cache/huggingface).

    Usage:
        client = LocalSDXLClient(output_dir=project_dir / "assets" / "raw" / "image")
        assets = client.generate("Karachi skyline at night, cinematic, 8K", num_images=2)
    """

    MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    _pipe = None  # class-level pipeline cache — shared across all instances

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Pipeline loading
    # ------------------------------------------------------------------

    def _load_pipeline(self):
        if LocalSDXLClient._pipe is not None:
            return LocalSDXLClient._pipe

        try:
            import torch  # noqa: PLC0415
            from diffusers import DiffusionPipeline  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "diffusers / torch not installed. Run:\n"
                "  pip install torch diffusers transformers accelerate"
            ) from exc

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        dtype = torch.float16
        logger.info("Loading SDXL pipeline on %s (dtype=%s)…", device, dtype)

        pipe = DiffusionPipeline.from_pretrained(
            self.MODEL_ID,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16",
        ).to(device)

        # Reduces peak memory on MPS — important for 24GB Mac configurations
        pipe.enable_attention_slicing()

        LocalSDXLClient._pipe = pipe
        logger.info("SDXL pipeline ready on %s", device)
        return pipe

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str, num_images: int = 2) -> list[Asset]:
        """Generate images locally. Returns list[Asset] with local_path already set.

        Args:
            prompt:     Full prompt string (use build_image_prompt() to build it).
            num_images: Number of images to generate per call (default 2).

        Returns:
            list[Asset] — each asset has local_path set (already saved to disk),
            quality_score=8.5, ready_for_use=True, source=GENERATED.
        """
        pipe = self._load_pipeline()

        logger.info("SDXL generating %d image(s) for: %r", num_images, prompt[:80])

        result = pipe(
            prompt=prompt,
            negative_prompt=_NEGATIVE_PROMPT,
            num_images_per_prompt=num_images,
            width=1344,              # SDXL optimal landscape — ~16:9
            height=768,
            num_inference_steps=20,  # fast; increase to 30 for higher quality
            guidance_scale=7.5,
        )

        assets: list[Asset] = []
        for img in result.images:
            asset_id = f"sdxl_{uuid.uuid4().hex[:8]}"
            dest = self.output_dir / f"{asset_id}.jpg"
            img.save(str(dest), format="JPEG", quality=95)

            assets.append(Asset(
                id=asset_id,
                type=AssetType.IMAGE,
                role=AssetRole.B_ROLL,
                source=AssetSource.GENERATED,
                source_id=asset_id,
                local_path=str(dest),
                filename=dest.name,
                resolution="1344x768",
                aspect_ratio="16:9",
                # High baseline so module_4 orchestrator prefers AI images over
                # low-scoring stock photos
                quality_score=8.5,
                ready_for_use=True,
                licensing=LicensingInfo(
                    license_type="SDXL Generated — royalty free",
                    attribution_required=False,
                    commercial_use=True,
                ),
                attribution="Generated locally via Stable Diffusion XL",
                search_query=prompt,
                visual_tags=["generated", "cinematic", "ai", "photorealistic"],
            ))
            logger.info("SDXL saved: %s", dest)

        return assets
