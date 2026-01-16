from __future__ import annotations
import os
import sys
import logging
import warnings
from pathlib import Path
from platform import system
from datetime import date

import pandas as pd
from dotenv import load_dotenv

pd.set_option("display.max_columns", None)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

def _require_env(name: str) -> str:
    v = os.getenv(name)
    if v is None or v == "":
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Set it in your .env file."
        )
    return v

def _get_env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, None)
    return default if (v is None or v == "") else v

def _to_bool(x: str) -> bool:
    return x.strip().lower() in {"1", "true", "yes", "on", "t", "y"}

def _to_int(x: str) -> int:
    return int(x.strip())

def if_relative_make_abs(path: str | Path) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (BASE_DIR / p).resolve()

def get_os() -> str:
    name = system()
    if name == "Windows":
        return "windows"
    elif name in {"Darwin", "Linux"}:
        return "nix"
    return "unknown"

FRED_API_KEY    = _require_env("FRED_API_KEY")
DATA_PATH       = if_relative_make_abs(_require_env("DATA_PATH"))
START_DATE      = _require_env("START_DATE")      
END_DATE        = date.today()     
BACKTEST_MONTHS = _to_int(_require_env("BACKTEST_MONTHS"))
DELIM          = _get_env("DELIM", ".")
DEBUG          = _to_bool(_get_env("DEBUG", "False"))
REFRESH_CACHE  = _to_bool(_get_env("REFRESH_CACHE", "False"))
OUTPUT_DIR     = if_relative_make_abs(_get_env("OUTPUT_DIR", "_output"))
TEMP_DIR       = if_relative_make_abs(_get_env("TEMP_DIR", "./_temp"))
UNIFIER_USER   = _require_env("UNIFIER_USER")
UNIFIER_TOKEN  = _require_env("UNIFIER_TOKEN")
MODEL_TYPE = _get_env("MODEL_TYPE")
WARM_START_INTERVAL = _to_int(_get_env("WARM_START_INTERVAL", "3"))

CACHE_DATA_DIR = DATA_PATH / "cache"
BACKTEST_DIR   = OUTPUT_DIR / "backtest"

def setup_logger(script_path: str, temp_dir: Path) -> logging.Logger:
    script_name = Path(script_path).stem
    log_file_path = temp_dir / f"{script_name}_logger.log"
    error_log_file_path = temp_dir / f"{script_name}_errors.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        return logger

    fmt = "%(asctime)s - %(levelname)s - %(name)s - %(filename)s - %(funcName)s:%(lineno)d - %(message)s"
    formatter = logging.Formatter(fmt)

    fh = logging.FileHandler(log_file_path);  fh.setLevel(logging.INFO);  fh.setFormatter(formatter)
    ch = logging.StreamHandler(sys.stdout);   ch.setLevel(logging.INFO);  ch.setFormatter(formatter)
    eh = logging.FileHandler(error_log_file_path); eh.setLevel(logging.ERROR); eh.setFormatter(formatter)

    logger.addHandler(ch); logger.addHandler(fh); logger.addHandler(eh)

    for third_party in ["urllib3", "pandas", "statsmodels"]:
        logging.getLogger(third_party).setLevel(logging.WARNING)

    logging.captureWarnings(True)
    return logger

def create_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    create_dirs()
