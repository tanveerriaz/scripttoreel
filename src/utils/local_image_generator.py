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
    "thriller":    "cinematic wide shot, dramatic lighting, film noir, 8K, photorealistic, high contrast",
    "horror":      "dark atmospheric wide shot, moonlit environment, eerie mist, horror movie still, photorealistic",
    "action":      "dynamic wide composition, motion blur, cinematic action establishing shot, golden hour, high contrast",
    "spy":         "espionage thriller, wide covert environment, night scene, cinematic, photorealistic",
    "heist":       "cinematic heist wide shot, tense urban atmosphere, night, photorealistic",
    "documentary": "documentary photography, natural light, wide establishing shot, real location, 8K, photorealistic",
    "educational": "clean, well-lit, wide shot, professional setting, informative, photorealistic",
    "birthday":    "vibrant celebration wide shot, bokeh background, warm golden lighting, professional photography",
    "celebration": "joyful, colourful, wide establishing shot, festive atmosphere, warm lighting, photorealistic",
    "wedding":     "elegant wide venue shot, soft bokeh, golden hour, romantic, professional wedding photography",
    "science":     "laboratory wide shot, high-tech equipment, clean scientific aesthetic, photorealistic",
    "nature":      "National Geographic style, wide panorama, golden hour, dramatic landscape, photorealistic",
    "travel":      "travel photography, wide establishing shot, iconic landmark, golden hour, cinematic",
    "business":    "corporate professional, wide office or cityscape, modern architecture, business magazine style, photorealistic",
    "food":        "food photography, shallow depth of field, moody restaurant lighting, wide table shot, appetizing",
    "history":     "historical wide shot, architecture, archival aesthetic, dramatic lighting, photorealistic",
    # Tech/coding topics — dark IDE, screen glow, abstract data visuals
    "coding":      "developer workspace dark mode IDE on monitor, code syntax highlighting, screen glow, cinematic wide shot, photorealistic",
    "technology":  "modern tech workspace wide shot, multiple monitors with code, dark ambient lighting, cinematic, photorealistic",
    "software":    "developer workspace dark mode IDE on monitor, code syntax highlighting, screen glow, cinematic wide shot, photorealistic",
    "ai":          "abstract neural network visualization glowing nodes, data streams, dark background, wide cinematic, photorealistic",
    "programming": "developer at desk dark mode code editor glowing screen, wide establishing shot, cinematic, photorealistic",
    "developer":   "software developer workspace wide shot, multiple screens code editor, dark theme, cinematic lighting, photorealistic",
}

# Wide establishing shots are safer than close-up portraits across ALL topics:
# SDXL reliably renders landscapes, architecture, and environments at high quality,
# whereas close-up human faces and hands often have anatomy distortions.
_DEFAULT_MODIFIER = (
    "wide establishing shot, cinematic photography, "
    "professional lighting, 8K, photorealistic, high detail"
)

# CLIP tokenizer hard limit is 77 tokens — keep this under 70 to leave headroom.
# Priority order: content safety first, anatomy artifacts second, quality last.
_NEGATIVE_PROMPT = (
    "cartoon, 3d render, anime, watermark, text, nude, nsfw, "
    "blurry, low quality, deformed, bad anatomy, "
    "deformed face, asymmetric eyes, extra fingers, mutated hands, bad hands, "
    "ai-generated look, CGI, digital rendering, over-saturated, plastic skin, over-sharpened"
)


def build_image_prompt(query: str, topic: str, tone: str = "") -> str:
    """Build an enhanced SDXL prompt from a stock search query + topic context.

    Detects genre/tone keywords and appends matching cinematic style modifiers.
    Longer/more-specific keywords take priority (sorted by length descending)
    so 'documentary' doesn't swallow 'pakistan' on a Pakistan travel topic.

    Example:
        query="night chase rooftop Karachi", topic="Pakistani action thriller"
        → "night chase rooftop Karachi, cinematic still, dramatic lighting, film noir, 8K, photorealistic"
    """
    combined = f"{topic} {tone}".lower()
    modifier = _DEFAULT_MODIFIER
    # Sort by keyword length (longest first) so specific terms like "pakistan"
    # win over generic ones like "travel" or "documentary"
    for keyword, mod in sorted(_GENRE_MODIFIERS.items(), key=lambda x: -len(x[0])):
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
            import diffusers as _diffusers          # noqa: PLC0415
            import transformers as _transformers    # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "diffusers / torch not installed. Run:\n"
                "  pip install torch diffusers transformers accelerate"
            ) from exc

        # Suppress loading-bar spam and tokenizer warnings from diffusers/transformers
        _diffusers.utils.logging.set_verbosity_error()
        _transformers.logging.set_verbosity_error()

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # MPS + float16 is unreliable: NaN values propagate through UNet attention
        # layers and corrupt latents before they reach the VAE decoder, producing
        # blank/black output images. Use float32 on MPS for stable inference.
        # CUDA can safely use float16 for performance.
        if device == "mps":
            dtype = torch.float32
            load_kwargs = {"torch_dtype": dtype, "use_safetensors": True}
        else:
            dtype = torch.float16
            load_kwargs = {"torch_dtype": dtype, "use_safetensors": True, "variant": "fp16"}

        logger.info("Loading SDXL pipeline on %s (dtype=%s)…", device, dtype)

        pipe = DiffusionPipeline.from_pretrained(
            self.MODEL_ID,
            **load_kwargs,
        ).to(device)

        # Reduces peak memory — important on MPS (unified memory architecture)
        pipe.enable_attention_slicing()

        # Silence diffusers' own tqdm step-by-step progress output
        pipe.set_progress_bar_config(disable=True)

        # MPS warmup pass — the first inference on MPS initialises Metal compute
        # shaders and JIT-compiles kernels. Without this, the very first real
        # image is often blank or corrupted. One cheap step with output_type="latent"
        # primes the pipeline without saving any output.
        if device == "mps":
            logger.info("SDXL: running MPS warmup pass…")
            import warnings as _warnings  # noqa: PLC0415
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                pipe(
                    "warmup",
                    num_inference_steps=1,
                    output_type="latent",
                    width=64,
                    height=64,
                )
            logger.info("SDXL: MPS warmup complete")

        LocalSDXLClient._pipe = pipe
        logger.info("SDXL pipeline ready on %s", device)
        return pipe

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str, num_images: int = 2, num_steps: int = 28) -> list[Asset]:
        """Generate images locally. Returns list[Asset] with local_path already set.

        Args:
            prompt:     Full prompt string (use build_image_prompt() to build it).
            num_images: Number of images to generate per call (default 2).
            num_steps:  Denoising steps — 20 for speed (short videos), 28 for quality.

        Returns:
            list[Asset] — each asset has local_path set (already saved to disk),
            quality_score=8.5, ready_for_use=True, source=GENERATED.
        """
        pipe = self._load_pipeline()

        logger.info(
            "SDXL generating %d image(s) @ %d steps for: %r",
            num_images, num_steps, prompt[:80],
        )

        # Suppress diffusers/transformers warnings during inference
        import warnings  # noqa: PLC0415
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")

        result = pipe(
            prompt=prompt,
            negative_prompt=_NEGATIVE_PROMPT,
            num_images_per_prompt=num_images,
            width=1344,              # SDXL optimal landscape — ~16:9
            height=768,
            num_inference_steps=num_steps,
            guidance_scale=8.5,
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
