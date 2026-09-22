"""Typed configuration: dataclass defaults overridden by config.yaml.

Defaults live in code so the app runs with no config file at all; config.yaml
only carries overrides. Unknown keys are rejected loudly rather than ignored,
because a silently-misspelled key is the classic source of "why is my setting
not doing anything".
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class AudioCfg:
    source: str = "loopback"          # loopback | mic
    device: str | None = None
    backend: str = "pyaudiowpatch"    # pyaudiowpatch | soundcard


@dataclass(slots=True)
class VadCfg:
    model: str = r"D:\AI\Models\silero_vad.onnx"
    threshold: float = 0.5
    endpoint_silence_ms: int = 550
    eager_punct_ms: int = 250
    min_speech_ms: int = 300
    max_segment_s: float = 8.0
    preroll_ms: int = 300


@dataclass(slots=True)
class AsrCfg:
    model: str = r"D:\AI\Models\whisper-large-v3-turbo-int8-ov"
    device: str = "GPU"
    language: str = "en"
    # Off by default: measured (tools/measure_power.py A/B) that re-running
    # Whisper every 1.5 s during speech doubles iGPU power, 2.46 W -> 5.02 W,
    # to render a dim English preview line. Set to 1500 if you want it back.
    partial_interval_ms: int = 0
    drop_hallucinations: bool = True


@dataclass(slots=True)
class MtCfg:
    model: str = r"D:\AI\Models\Qwen3-4B-int4-ov"
    fast_model: str = r"D:\AI\Models\Qwen2.5-1.5B-Instruct-int4-ov"
    device: str = "GPU"
    max_new_tokens: int = 200
    context_pairs: int = 3
    chat_reset_every: int = 8
    fallback_on_backlog: int = 4


@dataclass(slots=True)
class RuntimeCfg:
    gpu_concurrency: str = "serial"   # E1-measured: parallel is 4.69x worse
    cache_dir: str = ".ovcache"
    cache_budget_gb: float = 8.0      # LRU-evicted; see app/storage.py
    log_level: str = "info"
    metrics: bool = True


@dataclass(slots=True)
class GlossaryCfg:
    path: str = "glossary.tsv"
    enabled: bool = True


@dataclass(slots=True)
class UiCfg:
    opacity: float = 0.85
    font_size: int = 24
    max_lines: int = 3
    show_source: bool = True
    click_through: bool = True


@dataclass(slots=True)
class Config:
    audio: AudioCfg = field(default_factory=AudioCfg)
    vad: VadCfg = field(default_factory=VadCfg)
    asr: AsrCfg = field(default_factory=AsrCfg)
    mt: MtCfg = field(default_factory=MtCfg)
    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)
    glossary: GlossaryCfg = field(default_factory=GlossaryCfg)
    ui: UiCfg = field(default_factory=UiCfg)

    root: Path = field(default_factory=Path.cwd)

    # -- derived helpers -------------------------------------------------
    @property
    def cache_path(self) -> Path:
        p = Path(self.runtime.cache_dir)
        return p if p.is_absolute() else self.root / p

    @property
    def glossary_path(self) -> Path:
        p = Path(self.glossary.path)
        return p if p.is_absolute() else self.root / p

    def ov_props(self) -> dict[str, Any]:
        """Properties handed to every OpenVINO pipeline.

        CACHE_DIR measured (E2): both models 12.9s cold -> 3.4s warm.
        """
        return {"CACHE_DIR": str(self.cache_path), "PERFORMANCE_HINT": "LATENCY"}


def _fill(cls, data: dict[str, Any], where: str):
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown key(s) in '{where}': {sorted(unknown)}")
    return cls(**data)


def load(path: str | Path | None = None, root: Path | None = None) -> Config:
    """Load config.yaml over the dataclass defaults."""
    root = Path(root or Path.cwd()).resolve()
    cfg = Config(root=root)
    if path is None:
        path = root / "config.yaml"
    path = Path(path)
    if not path.exists():
        return cfg

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sections = {f.name: f.type for f in dataclasses.fields(Config)
                if f.name != "root"}
    unknown = set(raw) - set(sections)
    if unknown:
        raise ValueError(f"unknown section(s) in {path.name}: {sorted(unknown)}")

    for name in sections:
        if name in raw and raw[name]:
            setattr(cfg, name, _fill(type(getattr(cfg, name)), raw[name], name))

    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if cfg.audio.source not in ("loopback", "mic"):
        raise ValueError(f"audio.source must be loopback|mic, got {cfg.audio.source!r}")
    if cfg.audio.backend not in ("pyaudiowpatch", "soundcard"):
        raise ValueError(f"audio.backend must be pyaudiowpatch|soundcard, "
                         f"got {cfg.audio.backend!r}")
    if cfg.runtime.gpu_concurrency not in ("serial", "parallel"):
        raise ValueError("runtime.gpu_concurrency must be serial|parallel")
    if not 0 < cfg.vad.threshold < 1:
        raise ValueError("vad.threshold must be in (0, 1)")
    if cfg.vad.endpoint_silence_ms < 100:
        raise ValueError("vad.endpoint_silence_ms below 100 will chop every word")
    for label, p in (("asr.model", cfg.asr.model), ("mt.model", cfg.mt.model),
                     ("vad.model", cfg.vad.model)):
        if not Path(p).exists():
            raise FileNotFoundError(f"{label} not found: {p}")
