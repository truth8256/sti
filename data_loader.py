# =============================
# File: data_loader.py
# =============================
from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Union

# ---------- Internal CSV readers ----------

def _read_csv_safe(path: Path,
                   encoding_order: List[str] = ["utf-8", "cp949"],
                   dtype: Optional[Dict[str, Union[str, type]]] = None) -> pd.DataFrame:
    """
    단일 경로에서 인코딩 우선순위를 바꿔가며 안전하게 읽기.
    실패 시 빈 DataFrame 반환.
    """
    if not path.exists():
        return pd.DataFrame()
    for enc in encoding_order:
        try:
            return pd.read_csv(path, encoding=enc, dtype=dtype)
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    return pd.DataFrame()


def _read_csv_safe_any(paths: List[Path],
                       dtype: Optional[Dict[str, Union[str, type]]] = None) -> pd.DataFrame:
    """
    여러 경로 후보를 순서대로 시도해서 첫 성공 DataFrame 반환.
    실패 시 빈 DataFrame.
    """
    for p in paths:
        df = _read_csv_safe(p, dtype=dtype)
        if not df.empty:
            return df
    return pd.DataFrame()


# ---------- Light post-processing helpers ----------

def _tidy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    컬럼명 앞뒤 공백 제거 및 중복 컬럼 처리.
    """
    if df is None or df.empty:
        return df
    # strip + dedupe
    cols = []
    seen = {}
    for c in df.columns:
        base = c.strip()
        if base not in seen:
            seen[base] = 0
            cols.append(base)
        else:
            seen[base] += 1
            cols.append(f"{base}.{seen[base]}")
    df.columns = cols
    return df


def _ensure_str(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    코드/식별자 컬럼 문자열 보존.
    """
    if df is None or df.empty:
        return df
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


# ---------- Public loaders (7 files) ----------

def load_bookmark(data_dir: Path) -> pd.DataFrame:
    """
    bookmark.csv: 파일명과 해당 헤더 목록이 정리되어 있는 매핑 테이블.
    추후 스키마 검증/리네임 정책에 활용 가능.
    """
    df = _read_csv_safe_any([
        data_dir / "bookmark.csv",
        Path("/mnt/data") / "bookmark.csv"
    ])
    return _tidy_columns(df)


def load_population_agg(data_dir: Path) -> pd.DataFrame:
    """
    population.csv: (동 단위 원자료 또는 집계본)
    - downstream에서 구 단위로 합산할 수 있도록 '코드' 등 문자열 보존
    """
    df = _read_csv_safe_any([
        data_dir / "population.csv",
        Path("/mnt/data") / "population.csv"
    ])
    df = _tidy_columns(df)
    df = _ensure_str(df, ["코드", "지역구", "구", "동"])
    return df


def load_party_labels(data_dir: Path) -> pd.DataFrame:
    """
    party_labels.csv: 정당 코드-라벨 매핑
    """
    df = _read_csv_safe_any([
        data_dir / "party_labels.csv",
        Path("/mnt/data") / "party_labels.csv"
    ])
    df = _tidy_columns(df)
    # 흔히 쓰일만한 키 후보들 문자열화
    df = _ensure_str(df, ["정당코드", "정당", "party", "code"])
    return df


def load_vote_trend(data_dir: Path) -> pd.DataFrame:
    """
    vote_trend.csv: 정당 성향별/정당별 득표 추이
    """
    df = _read_csv_safe_any([
        data_dir / "vote_trend.csv",
        Path("/mnt/data") / "vote_trend.csv"
    ])
    df = _tidy_columns(df)
    df = _ensure_str(df, ["코드", "선거구명", "지역구", "district", "label"])
    return df


def load_results_2024(data_dir: Path) -> pd.DataFrame:
    """
    5_na_dis_results.csv: 2024 총선 결과(동/선거구 레벨)
    """
    df = _read_csv_safe_any([
        data_dir / "5_na_dis_results.csv",
        Path("/mnt/data") / "5_na_dis_results.csv"
    ])
    df = _tidy_columns(df)
    df = _ensure_str(df, ["코드", "선거구명", "지역구", "구", "동"])
    return df


def load_current_info(data_dir: Path) -> pd.DataFrame:
    """
    current_info.csv: 현직/주요 인물/현황 정보
    """
    df = _read_csv_safe_any([
        data_dir / "current_info.csv",
        Path("/mnt/data") / "current_info.csv"
    ])
    df = _tidy_columns(df)
    df = _ensure_str(df, ["코드", "선거구명", "지역구", "이름", "정당"])
    return df


def load_index_sample(data_dir: Path) -> pd.DataFrame:
    """
    index_sample1012.csv (선택): 지표/스코어 샘플
    """
    df = _read_csv_safe_any([
        data_dir / "index_sample1012.csv",
        Path("/mnt/data") / "index_sample1012.csv"
    ])
    df = _tidy_columns(df)
    df = _ensure_str(df, ["코드", "선거구명", "지역구"])
    return df


# ---------- Optional: convenience aggregator ----------

def load_all(data_dir: Union[str, Path]) -> dict:
    """
    대시보드에서 한 번에 호출할 수 있는 일괄 로더.
    """
    data_dir = Path(data_dir)
    return {
        "bookmark": load_bookmark(data_dir),
        "population": load_population_agg(data_dir),
        "party_labels": load_party_labels(data_dir),
        "vote_trend": load_vote_trend(data_dir),
        "results_2024": load_results_2024(data_dir),
        "current_info": load_current_info(data_dir),
        "index_sample": load_index_sample(data_dir),
    }
