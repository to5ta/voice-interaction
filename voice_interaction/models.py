"""Model download and discovery for all three engines.

Piper voices are two loose files (.onnx + .onnx.json) from HuggingFace, Vosk
models are zip archives from alphacephei.com, and the chat mode's LLM is a
single GGUF file, again from HuggingFace. All are fetched with urllib so the
install has no extra dependency just to download things."""

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from . import config as cfg

PIPER_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
VOSK_BASE_URL = "https://alphacephei.com/vosk/models"
HF_BASE_URL = "https://huggingface.co"


def _report(block_num, block_size, total_size):
    """Progress on one rewritten line — suppressed when stdout is redirected,
    where \\r produces a useless wall of text."""
    if total_size <= 0 or not sys.stdout.isatty():
        return
    done = min(block_num * block_size, total_size)
    pct = int(100 * done / total_size)
    if pct == _report.last_pct and done < total_size:
        return
    _report.last_pct = pct
    sys.stdout.write(f"\r  {pct:3d}%  ({done // 1024 // 1024} / {total_size // 1024 // 1024} MB)")
    sys.stdout.flush()
    if done >= total_size:
        sys.stdout.write("\n")


_report.last_pct = -1


def _piper_url_path(voice: str) -> str:
    # "de_DE-thorsten-medium" -> de/de_DE/thorsten/medium/de_DE-thorsten-medium
    lang_region, name, quality = voice.split("-")
    lang = lang_region.split("_")[0]
    return f"{lang}/{lang_region}/{name}/{quality}/{voice}"


def download_piper_voice(voice: str, force: bool = False) -> Path:
    cfg.PIPER_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path, config_path = cfg.piper_paths(voice)
    base = _piper_url_path(voice)

    for suffix, target in ((".onnx", onnx_path), (".onnx.json", config_path)):
        if target.exists() and not force:
            print(f"  have {target.name}")
            continue
        url = f"{PIPER_BASE_URL}/{base}{suffix}"
        print(f"  downloading {target.name}")
        urllib.request.urlretrieve(url, target, reporthook=_report)

    return onnx_path


def download_vosk_model(model: str, force: bool = False) -> Path:
    cfg.VOSK_DIR.mkdir(parents=True, exist_ok=True)
    target = cfg.vosk_path(model)

    if target.exists() and not force:
        print(f"  have {model}/")
        return target
    if target.exists():
        shutil.rmtree(target)

    archive = cfg.VOSK_DIR / f"{model}.zip"
    url = f"{VOSK_BASE_URL}/{model}.zip"
    print(f"  downloading {model}.zip")
    urllib.request.urlretrieve(url, archive, reporthook=_report)

    print(f"  extracting {model}.zip")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(cfg.VOSK_DIR)
    archive.unlink()

    if not target.exists():
        raise RuntimeError(f"archive did not contain expected directory {target}")
    return target


def download_llm(name: str = None, force: bool = False) -> Path:
    """Fetch one GGUF model for the chat mode."""
    model = cfg.get_llm_model(name)
    cfg.LLM_DIR.mkdir(parents=True, exist_ok=True)
    target = cfg.llm_path(model)

    if target.exists() and not force:
        print(f"  have {target.name}")
        return target

    url = f"{HF_BASE_URL}/{model.repo}/resolve/main/{model.filename}"
    print(f"{model.name} ({model.size_mb} MB):")
    print(f"  downloading {model.filename}")
    # Download beside the target and rename, so an interrupted transfer never
    # leaves a truncated GGUF that llama.cpp would later fail to parse.
    partial = target.with_suffix(target.suffix + ".part")
    urllib.request.urlretrieve(url, partial, reporthook=_report)
    partial.replace(target)
    return target


def missing_llm(name: str = None) -> list:
    model = cfg.get_llm_model(name)
    if cfg.llm_path(model).exists():
        return []
    return [f"LLM '{model.name}'"]


def require_llm(name: str = None) -> None:
    model = cfg.get_llm_model(name)
    if missing_llm(model.name):
        raise SystemExit(
            f"Missing model for chat: {model.filename} ({model.size_mb} MB).\n"
            f"Run: python -m voice_interaction download --llm {model.name}"
        )


def ensure_language(code: str, force: bool = False) -> None:
    """Download everything needed to both understand and speak one language."""
    lang = cfg.get_language(code)
    print(f"{lang.label} ({lang.code}):")
    download_vosk_model(lang.vosk_model, force=force)
    download_piper_voice(lang.piper_voice, force=force)


def missing_for(code: str) -> list:
    """Which model files are absent for a language — used to fail with a
    helpful message instead of a stack trace deep inside an engine."""
    lang = cfg.get_language(code)
    missing = []
    if not cfg.vosk_path(lang.vosk_model).exists():
        missing.append(f"Vosk model '{lang.vosk_model}'")
    onnx_path, config_path = cfg.piper_paths(lang.piper_voice)
    if not onnx_path.exists() or not config_path.exists():
        missing.append(f"Piper voice '{lang.piper_voice}'")
    return missing


def require_language(code: str) -> None:
    missing = missing_for(code)
    if missing:
        raise SystemExit(
            f"Missing model(s) for '{code}': {', '.join(missing)}.\n"
            f"Run: python -m voice_interaction download --lang {code}"
        )
