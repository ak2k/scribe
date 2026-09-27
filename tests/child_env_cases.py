"""Environment names the uvx children must and must not be given, shared by both runners' tests."""

from __future__ import annotations

# Keys scribe's own backends read, a key file's path, what only a `uv run`
# parent sets, a file uv would load into the tool's environment, and for each
# word that marks a credential, a name holding it under a prefix whose other
# names pass.
DROPPED = (
    "XAI_API_KEY",
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
    "HONCHO_API_KEY",
    "SOPS_AGE_KEY_FILE",
    "VIRTUAL_ENV",
    "UV_ENV_FILE",
    "UV_PUBLISH_TOKEN",
    "UV_INDEX_CORP_PASSWORD",
    "HF_X_SECRET",
    "UV_API_KEY",
    "MLX_X_CREDENTIAL",
    "PYTORCH_X_CREDENTIALS",
    # Without the ID token it goes with, Hugging Face's client fails every call.
    "HF_OIDC_RESOURCE",
)
# Where the worker alone looks for the Hugging Face token.
TOKENS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH")
# Every name the children are given by name, every proxy name in either case,
# and one name under each prefix whose names pass.
KEPT = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "TMPDIR",
    "TEMP",
    "TMP",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    "XDG_CONFIG_HOME",
    "XDG_CONFIG_DIRS",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "HTTP_TIMEOUT",
    "TRANSFORMERS_OFFLINE",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN",
    "UV_NO_HF_TOKEN",
    "DO_NOT_TRACK",
    "DISABLE_TELEMETRY",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
    "PARAKEET_CACHE_DIR",
    "http_proxy",
    "HTTP_PROXY",
    "https_proxy",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
    "no_proxy",
    "NO_PROXY",
    "LC_ALL",
    "UV_CACHE_DIR",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "MLX_METAL_FAST_SYNCH",
    "PYTORCH_ENABLE_MPS_FALLBACK",
)
