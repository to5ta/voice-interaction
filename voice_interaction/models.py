"""Model download and discovery for both engines.

Piper voices are two loose files (.onnx + .onnx.json) from HuggingFace;
Vosk models are zip archives from alphacephei.com. Both are fetched with
urllib so the install has no extra dependency just to download things."""

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from . import config as cfg

PIPER_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
VOSK_BASE_URL = "https://alphacephei.com/vosk/models"


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
