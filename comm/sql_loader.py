# comm/sql_loader.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple

class SqlRepo:
    """
    MyBatis 느낌으로 'store.select_tb_rotation' 같은 ID로
    sql/store/select_tb_rotation.sql 파일을 찾아서 읽어옵니다.
    파일 mtime을 캐시해 재로드 비용을 줄입니다.
    """
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir).resolve()
        self._cache: Dict[str, Tuple[float, str]] = {}  # path -> (mtime, text)

    def _load_file(self, path: Path) -> str:
        key = str(path)
        mtime = path.stat().st_mtime
        cached = self._cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1]
        text = path.read_text(encoding="utf-8")
        self._cache[key] = (mtime, text)
        return text

    def get(self, mapper_id: str) -> str:
        """
        'store.select_tb_rotation' -> sql/store/select_tb_rotation.sql
        """
        rel = Path(*mapper_id.split(".")).with_suffix(".sql")
        path = (self.base_dir / rel).resolve()
        if not path.exists():
            raise FileNotFoundError(f"SQL not found for id '{mapper_id}' at {path}")
        return self._load_file(path)
