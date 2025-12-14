# ComfyUI-QwenVL
# This custom node integrates the Qwen-VL series, including the latest Qwen3-VL models,
# including Qwen2.5-VL and the latest Qwen3-VL, to enable advanced multimodal AI for text generation,
# image understanding, and video analysis.
#
# Models License Notice:
# - Qwen3-VL: Apache-2.0 License (https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
# - Qwen2.5-VL: Apache-2.0 License (https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
#
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/1038lab/ComfyUI-QwenVL
# Modified for ZLUDA compatibility - bitsandbytes disabled

import gc
import json
from enum import Enum
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

import folder_paths

NODE_DIR = Path(__file__).parent
CONFIG_PATH = NODE_DIR / "config.json"
MODEL_CONFIGS = {}
SYSTEM_PROMPTS = {}


TOOLTIPS = {
    "model_name": "Pick the Qwen-VL checkpoint. First run downloads weights into models/LLM/Qwen-VL, so leave disk space.",
    "quantization": "Precision vs VRAM. FP16 gives the best quality if memory allows. Note: Quantization disabled for ZLUDA compatibility.",
    "attention_mode": "auto tries flash-attn v2 when installed and falls back to SDPA. Only override when debugging attention backends.",
    "preset_prompt": "Built-in instruction describing how Qwen-VL should analyze the media input.",
    "custom_prompt": "Optional override—when filled it completely replaces the preset template.",
    "max_tokens": "Maximum number of new tokens to decode. Larger values yield longer answers but consume more time and memory.",
    "keep_model_loaded": "Keeps the model resident in VRAM/RAM after the run so the next prompt skips loading.",
    "seed": "Seed controlling sampling and frame picking; reuse it to reproduce results.",
    "use_torch_compile": "Enable torch.compile('reduce-overhead') on supported CUDA/Torch 2.1+ builds for extra throughput after the first compile.",
    "device": "Force execution on cuda/cpu/mps or leave auto to follow hardware detection.",
    "temperature": "Sampling randomness when num_beams == 1. 0.2–0.4 is focused, 0.7+ is creative.",
    "top_p": "Nucleus sampling cutoff when num_beams == 1. Lower values keep only top tokens; 0.9–0.95 allows more variety.",
    "num_beams": "Beam-search width. Values >1 disable temperature/top_p and trade speed for more stable answers.",
    "repetition_penalty": "Values >1 (e.g., 1.1–1.3) penalize repeated phrases; 1.0 leaves logits untouched.",
    "frame_count": "Number of frames extracted from video inputs before prompting Qwen-VL. More frames provide context but cost time.",
}


class Quantization(str, Enum):
    # Quantization disabled for ZLUDA - only FP16 supported
    FP16 = "None (FP16)"

    @classmethod
    def get_values(cls):
        return [item.value for item in cls]

    @classmethod
    def from_value(cls, value):
        # Always return FP16 for ZLUDA compatibility
        return cls.FP16


ATTENTION_MODES = ["auto", "flash_attention_2", "sdpa"]


def load_model_configs():
    global MODEL_CONFIGS, SYSTEM_PROMPTS
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            MODEL_CONFIGS = json.load(fh)
            SYSTEM_PROMPTS = MODEL_CONFIGS.get("_system_prompts", {})
    except Exception as exc:
        print(f"[QwenVL] Config load failed: {exc}")
        MODEL_CONFIGS = {}
        SYSTEM_PROMPTS = {}
    custom = NODE_DIR / "custom_models.json"
    if custom.exists():
        try:
            with open(custom, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
            models = data.get("hf_models", {}) or data.get("models", {})
            if models:
                MODEL_CONFIGS.update(models)
                print(f"[QwenVL] Loaded {len(models)} custom models")
        except Exception as exc:
            print(f"[QwenVL] custom_models.json skipped: {exc}")


if not MODEL_CONFIGS:
    load_model_configs()


def get_device_info():
    gpu = {"available": False, "total_memory": 0, "free_memory": 0}
    device_type = "cpu"
    recommended = "cpu"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1024**3
        gpu = {
            "available": True,
            "total_memory": total,
            "free_memory": total - (torch.cuda.memory_allocated(0) / 1024**3),
        }
        device_type = "nvidia_gpu"
        recommended = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device_type = "apple_silicon"
        recommended = "mps"
        gpu = {"available": True, "total_memory": 0, "free_memory": 0}
    sys_mem = psutil.virtual_memory()
    return {
        "gpu": gpu,
        "system_memory": {
            "total": sys_mem.total / 1024**3,
            "available": sys_mem.available / 1024**3,
        },
        "device_type": device_type,
        "recommended_device": recommended,
    }


def flash_attn_available():
    if not torch.cuda.is_available():
        return False
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def resolve_attention_mode(mode):
    if mode == "sdpa":
        return "sdpa"
    if mode == "flash_attention_2":
        if flash_attn_available():
            return "flash_attention_2"
        print("[QwenVL] Flash-Attn forced but unavailable, falling back to SDPA")
        return "sdpa"
    if flash_attn_available():
        return "flash_attention_2"
    print("[QwenVL] Flash-Attn auto mode: dependency not ready, using SDPA")
    return "sdpa"


def ensure_model(model_name):
    info = MODEL_CONFIGS.get(model_name)
    if not info:
        raise ValueError(f"Model '{model_name}' not in config")
    repo_id = info["repo_id"]
    models_dir = Path(folder_paths.models_dir) / "LLM" / "Qwen-VL"
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / repo_id.split("/")[-1]
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md", ".git*"],
    )
    return str(target)


def enforce_memory(model_name, quantization, device_info):
    # For ZLUDA compatibility, always use FP16 (no quantization)
    print("[QwenVL-ZLUDA] Quantization disabled for ZLUDA compatibility - using FP16")
    return Quantization.FP16


def quantization_config(model_name, quantization):
    # Completely disable bitsandbytes for ZLUDA compatibility
    # Always return FP16 dtype
    info = MODEL_CONFIGS.get(model_name, {})
    if info.get("quantized"):
        print("[QwenVL-ZLUDA] Pre-quantized model detected, loading as-is")
        return None, None
    
    # Force FP16 for ZLUDA
    print("[QwenVL-ZLUDA] Loading model in FP16 (bitsandbytes disabled)")
    return None, torch.float16 if torch.cuda.is_available() else torch.float32


class QwenVLBase:
    def __init__(self):
        self.device_info = get_device_info()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_signature = None
        print(f"[QwenVL-ZLUDA] Node on {self.device_info['device_type']} (bitsandbytes disabled)")

    def clear(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_signature = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(
        self,
        model_name,
        quant_value,
        attention_mode,
        use_compile,
        device_choice,
        keep_model_loaded,
    ):
        # Force FP16 for ZLUDA
        quant = Quantization.FP16
        attn_impl = resolve_attention_mode(attention_mode)
        device = self.device_info["recommended_device"] if device_choice == "auto" else device_choice
        signature = (model_name, quant.value, attn_impl, device, use_compile)
        if keep_model_loaded and self.model is not None and self.current_signature == signature:
            return
        self.clear()
        model_path = ensure_model(model_name)
        quant_config, dtype = quantization_config(model_name, quant)
        load_kwargs = {
            "device_map": {"":  0} if device == "cuda" and torch.cuda.is_available() else device,
            "torch_dtype": dtype,
            "attn_implementation": attn_impl,
            "use_safetensors": True,
        }
        # Ensure bitsandbytes is never used
        if quant_config is not None:
            print("[QwenVL-ZLUDA] Warning: quantization_config ignored for ZLUDA compatibility")
        
        print(f"[QwenVL-ZLUDA] Loading {model_name} (FP16, attn={attn_impl})")
        self.model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs).eval()
        self.model.config.use_cache = True
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.use_cache = True
        if use_compile and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("[QwenVL-ZLUDA] torch.compile enabled")
            except Exception as exc:
                print(f"[QwenVL-ZLUDA] torch.compile skipped: {exc}")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.current_signature = signature

    @staticmethod
    def tensor_to_pil(tensor):
        if tensor is None:
            return None
        if tensor.dim() == 4:
            tensor = tensor[0]
        array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(array)

    @torch.no_grad()
    def generate(
        self,
        prompt_text,
        image,
        video,
        frame_count,
        max_tokens,
        temperature,
        top_p,
        num_beams,
        repetition_penalty,
    ):
        conversation = [{"role": "user", "content": []}]
        if image is not None:
            conversation[0]["content"].append({"type": "image", "image": self.tensor_to_pil(image)})
        if video is not None:
            frames = [self.tensor_to_pil(frame) for frame in video]
            if len(frames) > frame_count:
                idx = np.linspace(0, len(frames) - 1, frame_count, dtype=int)
                frames = [frames[i] for i in idx]
            if frames:
                conversation[0]["content"].append({"type": "video", "video": frames})
        conversation[0]["content"].append({"type": "text", "text": prompt_text})
        chat = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        images = [item["image"] for item in conversation[0]["content"] if item["type"] == "image"]
        video_frames = [frame for item in conversation[0]["content"] if item["type"] == "video" for frame in item["video"]]
        videos = [video_frames] if video_frames else None
        processed = self.processor(text=chat, images=images or None, videos=videos, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        model_inputs = {
            key: value.to(model_device) if torch.is_tensor(value) else value
            for key, value in processed.items()
        }
        stop_tokens = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, "eot_id") and self.tokenizer.eot_id is not None:
            stop_tokens.append(self.tokenizer.eot_id)
        kwargs = {
            "max_new_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "num_beams": num_beams,
            "eos_token_id": stop_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if num_beams == 1:
            kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
        else:
            kwargs["do_sample"] = False
        outputs = self.model.generate(**model_inputs, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        input_len = model_inputs["input_ids"].shape[-1]
        text = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
        return text.strip()

    def run(
        self,
        model_name,
        quantization,
        preset_prompt,
        custom_prompt,
        image,
        video,
        frame_count,
        max_tokens,
        temperature,
        top_p,
        num_beams,
        repetition_penalty,
        seed,
        keep_model_loaded,
        attention_mode,
        use_torch_compile,
        device,
    ):
        torch.manual_seed(seed)
        prompt = SYSTEM_PROMPTS.get(preset_prompt, preset_prompt)
        if custom_prompt and custom_prompt.strip():
            prompt = custom_prompt.strip()
        self.load_model(
            model_name,
            quantization,
            attention_mode,
            use_torch_compile,
            device,
            keep_model_loaded,
        )
        try:
            text = self.generate(
                prompt,
                image,
                video,
                frame_count,
                max_tokens,
                temperature,
                top_p,
                num_beams,
                repetition_penalty,
            )
            return (text,)
        finally:
            if not keep_model_loaded:
                self.clear()


class AILab_QwenVL(QwenVLBase):
    @classmethod
    def INPUT_TYPES(cls):
        models = [name for name in MODEL_CONFIGS.keys() if not name.startswith("_")]
        default_model = models[0] if models else "Qwen3-VL-4B-Instruct"
        prompts = MODEL_CONFIGS.get("_preset_prompts", ["Describe this image in detail."])
        preferred_prompt = "🖼️ Detailed Description"
        default_prompt = preferred_prompt if preferred_prompt in prompts else prompts[0]
        return {
            "required": {
                "model_name": (models, {"default": default_model, "tooltip": TOOLTIPS["model_name"]}),
                "quantization": (Quantization.get_values(), {"default": Quantization.FP16.value, "tooltip": TOOLTIPS["quantization"]}),
                "attention_mode": (ATTENTION_MODES, {"default": "auto", "tooltip": TOOLTIPS["attention_mode"]}),
                "preset_prompt": (prompts, {"default": default_prompt, "tooltip": TOOLTIPS["preset_prompt"]}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": TOOLTIPS["custom_prompt"]}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 2048, "tooltip": TOOLTIPS["max_tokens"]}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "tooltip": TOOLTIPS["keep_model_loaded"]}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1, "tooltip": TOOLTIPS["seed"]}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "🧪AILab/QwenVL"

    def process(
        self,
        model_name,
        quantization,
        preset_prompt,
        custom_prompt,
        attention_mode,
        max_tokens,
        keep_model_loaded,
        seed,
        image=None,
        video=None,
    ):
        return self.run(
            model_name,
            quantization,
            preset_prompt,
            custom_prompt,
            image,
            video,
            16,
            max_tokens,
            0.6,
            0.9,
            1,
            1.2,
            seed,
            keep_model_loaded,
            attention_mode,
            False,
            "auto",
        )


class AILab_QwenVL_Advanced(QwenVLBase):
    @classmethod
    def INPUT_TYPES(cls):
        models = [name for name in MODEL_CONFIGS.keys() if not name.startswith("_")]
        default_model = models[0] if models else "Qwen3-VL-4B-Instruct"
        prompts = MODEL_CONFIGS.get("_preset_prompts", ["Describe this image in detail."])
        preferred_prompt = "🖼️ Detailed Description"
        default_prompt = preferred_prompt if preferred_prompt in prompts else prompts[0]
        return {
            "required": {
                "model_name": (models, {"default": default_model, "tooltip": TOOLTIPS["model_name"]}),
                "quantization": (Quantization.get_values(), {"default": Quantization.FP16.value, "tooltip": TOOLTIPS["quantization"]}),
                "attention_mode": (ATTENTION_MODES, {"default": "auto", "tooltip": TOOLTIPS["attention_mode"]}),
                "use_torch_compile": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["use_torch_compile"]}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto", "tooltip": TOOLTIPS["device"]}),
                "preset_prompt": (prompts, {"default": default_prompt, "tooltip": TOOLTIPS["preset_prompt"]}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": TOOLTIPS["custom_prompt"]}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 4096, "tooltip": TOOLTIPS["max_tokens"]}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "tooltip": TOOLTIPS["temperature"]}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "tooltip": TOOLTIPS["top_p"]}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 8, "tooltip": TOOLTIPS["num_beams"]}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 2.0, "tooltip": TOOLTIPS["repetition_penalty"]}),
                "frame_count": ("INT", {"default": 16, "min": 1, "max": 64, "tooltip": TOOLTIPS["frame_count"]}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "tooltip": TOOLTIPS["keep_model_loaded"]}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1, "tooltip": TOOLTIPS["seed"]}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "🧪AILab/QwenVL"

    def process(
        self,
        model_name,
        quantization,
        attention_mode,
        use_torch_compile,
        device,
        preset_prompt,
        custom_prompt,
        max_tokens,
        temperature,
        top_p,
        num_beams,
        repetition_penalty,
        frame_count,
        keep_model_loaded,
        seed,
        image=None,
        video=None,
    ):
        return self.run(
            model_name,
            quantization,
            preset_prompt,
            custom_prompt,
            image,
            video,
            frame_count,
            max_tokens,
            temperature,
            top_p,
            num_beams,
            repetition_penalty,
            seed,
            keep_model_loaded,
            attention_mode,
            use_torch_compile,
            device,
        )


NODE_CLASS_MAPPINGS = {
    "AILab_QwenVL": AILab_QwenVL,
    "AILab_QwenVL_Advanced": AILab_QwenVL_Advanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_QwenVL": "QwenVL (ZLUDA)",
    "AILab_QwenVL_Advanced": "QwenVL Advanced (ZLUDA)",
}
