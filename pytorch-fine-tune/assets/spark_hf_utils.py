"""DGX Spark helpers for Hugging Face auth in fine-tuning scripts."""

import os

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError


def get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def check_model_access(model_name: str, hf_token: str | None) -> None:
    try:
        hf_hub_download(model_name, "config.json", token=hf_token)
    except GatedRepoError as exc:
        raise SystemExit(
            f"\nCannot access gated model '{model_name}'.\n"
            f"1. Open https://huggingface.co/{model_name} and request access while logged in as your HF user.\n"
            f"2. Ensure HF_TOKEN in ~/.config/secrets/hf_token is a token for that same account.\n"
            f"3. Re-run ./runme.sh after access is approved.\n\n"
            f"Original error: {exc}"
        ) from exc
