# comm/props_loader.py
from __future__ import annotations
from pathlib import Path
from typing import Dict

def load_properties(path: Path) -> Dict[str, str]:
    props: Dict[str, str] = {}
    p = Path(path).expanduser().resolve()
    with p.open(encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            # 주석/빈 줄
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            props[k.strip()] = v.strip()
    return props

def get_int(props: Dict[str, str], key: str, default: int) -> int:
    try:
        return int(props.get(key, default))
    except Exception:
        return default

def get_bool(props: Dict[str, str], key: str, default: bool = False) -> bool:
    return str(props.get(key, str(default))).strip().lower() in {"1","true","y","yes"}

def getHost(props: Dict[str, str]) -> str:
    return props.get("datasource.host")

def getPort(props: Dict[str, str]) -> int:
    return int(props.get("datasource.port"))

def getUser(props: Dict[str, str]) -> str:
    return props.get("datasource.user")

def getPassword(props: Dict[str, str]) -> str:
    return props.get("datasource.password")

def getSid(props: Dict[str, str]) -> str:
    return props.get("datasource.sid")

def getServiceName(props: Dict[str, str]) -> str:
    return props.get("datasource.service_name")
