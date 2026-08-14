import os

import torch
from diffusers import AutoPipelineForText2Image


MODEL_ID = "stabilityai/sd-turbo"
PROMPT = "A cute cat sitting on a rainbow."
OUTPUT_FILE = "cat_rainbow.png"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN before running this script."
        )

    device = get_device()
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32

    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        token=hf_token,
    )
    pipe = pipe.to(device)

    image = pipe(
        prompt=PROMPT,
        num_inference_steps=1,
        guidance_scale=0.0,
    ).images[0]

    image.save(OUTPUT_FILE)
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
