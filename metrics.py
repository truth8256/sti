# =============================
# File: metrics.py
# =============================
from __future__ import annotations

import re
import numpy as np
import pandas as pd

# 공통 코드 후보 (bookmark.csv 기준 다양성 대응)
_CODE_CANDIDATES = ["코드", "지역구코드", "선거구코드", "지역코드", "code", "CODE"]

def _canon_code(x: object) -> str:
    s = str(x).strip()
    s = re.sub(r"[^0-9A-Za-z]", "", s)
    s = s.lstrip("0")
    return s.lower()

def _pct_float(v) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().replace(",", "")
    m = re.match(r"^\s*([+-]?\d+(\.\d+)?)\s*%?\s*$", s)
    if not m:
        return None
    x = float(m.group(1))
    if "%" in s:
        return x
    return x * 100.0 if 0 <= x <= 1 else x

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df
    df2 = df.copy()
    df2.columns = [str(c).strip().replace("\n", "").replace("\r", "") for c in df2.columns]
    return df2

def _detect_code_col(df: pd.DataFrame) -> str | None:
    for c in _CODE_CANDIDATES:
        if c in df.columns:
            return c
    cols = [str(c).strip().replace("\n", "").replace("\r", "") for c in df.columns]
    for cand in _CODE_CANDIDATES:
        if cand in cols:
            return df.columns[cols.index(cand)]
    return None

def _get_by_code_local(df: pd.DataFrame, code: str) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df2 = _normalize_columns(df)
    col = "코드" if "코드" in df2.columns else _detect_code_col(df2)
    if not col:
        return pd.DataFrame()
    key = _canon_code(code)
    try:
        sub = df2[df2[col].astype(str).map(_canon_code) == key]
        return sub if len(sub) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def _extract_year_from_election(election: str) -> int | None:
    if not isinstance(election, str):
        return None
    m = re.match(r"^(\d{4})", election.strip())
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

def compute_trend_series(df_trend: pd.DataFrame, code: str) -> pd.DataFrame:
    """
    vote_trend.csv 전용:
    - long 형태(election/label/prop) → 연도 컬럼(year) 추가 후 반환
    - wide 형태(연도 + 계열 컬럼들)도 그대로 반환(차트에서 melt 처리)
    """
    sub = _get_by_code_local(df_trend, code)
    if sub.empty:
        return pd.DataFrame()
    sub = _normalize_columns(sub)

    # 우선 long 포맷 후보 탐지
    election_col = "election" if "election" in sub.columns else ("연도" if "연도" in sub.columns else None)
    label_col    = "label"    if "label"    in sub.columns else next((c for c in ["성향","정당성향","party_label"] if c in sub.columns), None)
    value_col    = "prop"     if "prop"     in sub.columns else next((c for c in ["득표율","비율","share","ratio","pct"] if c in sub.columns), None)

    # long 포맷 → 보정 후 반환
    if label_col and value_col:
        try:
            if sub[value_col].dtype == "O":
                sub[value_col] = sub[value_col].apply(_pct_float)
            elif pd.api.types.is_numeric_dtype(sub[value_col]):
                if (sub[value_col].dropna() <= 1).all():
                    sub[value_col] = sub[value_col] * 100.0
        except Exception:
            pass

        if election_col == "election":
            sub["year"] = sub["election"].apply(_extract_year_from_election)
        elif election_col == "연도":
            sub["year"] = pd.to_numeric(sub["연도"], errors="coerce")
        else:
            sub["year"] = pd.NA

        # long 그대로 반환(차트에서 직접 사용)
        return sub[["year", election_col] + ([label_col] if label_col else []) + ([value_col] if value_col else [])].dropna(subset=["year"]).reset_index(drop=True)

    # wide 포맷(연도 + 성향별 칼럼들) → 그대로 반환
    # 최소 요건: '연도' 또는 'year' 비슷한 축
    if "연도" in sub.columns:
        sub["year"] = pd.to_numeric(sub["연도"], errors="coerce")
    elif "year" not in sub.columns:
        # year가 없으면 추정 불가 → 그대로 반환
        return sub

    return sub

def compute_24_gap(df_24: pd.DataFrame, code: str) -> float | None:
    """
    2024(있으면 2024, 없으면 최신 연도)의 1~2위 득표율 격차(p).
    - 5_na_dis_results.csv 기준(후보1_득표율 / 후보2_득표율)
    - 호환: '1위득표율' / '2위득표율' 등도 지원
    """
    try:
        sub = _get_by_code_local(df_24, code)
        if sub.empty:
            return None

        # 2024 우선, 없으면 최신 연도
        if "연도" in sub.columns:
            tmp = sub.dropna(subset=["연도"]).copy()
            tmp["__year__"] = pd.to_numeric(tmp["연도"], errors="coerce")
            if (tmp["__year__"] == 2024).any():
                row = tmp[tmp["__year__"] == 2024].iloc[0]
            else:
                row = tmp.loc[tmp["__year__"].idxmax()]
        else:
            row = sub.iloc[0]

        c1v = next((c for c in ["후보1_득표율","1위득표율","1위 득표율","1st_share","득표율_1위","1위득표율(%)"] if c in sub.columns), None)
        c2v = next((c for c in ["후보2_득표율","2위득표율","2위 득표율","2nd_share","득표율_2위","2위득표율(%)"] if c in sub.columns), None)

        if not (c1v and c2v):
            return None

        v1 = _pct_float(row[c1v])
        v2 = _pct_float(row[c2v])
        if v1 is None or v2 is None:
            return None
        return round(v1 - v2, 2)
    except Exception:
        return None

def compute_summary_metrics(df_trend: pd.DataFrame, df_24: pd.DataFrame, df_idx: pd.DataFrame, code: str) -> dict:
    """
    index_sample1012.csv 기준:
    - PL_prg_str / PL_swing_B / PL_gap_B 사용
    - 누락 시 24년 격차 계산으로 보완
    """
    out = {"PL_prg_str": np.nan, "PL_swing_B": "N/A", "PL_gap_B": np.nan}

    sub = _get_by_code_local(df_idx, code)
    if sub is None or sub.empty:
        gap = compute_24_gap(df_24, code)
        if gap is not None:
            out["PL_gap_B"] = gap
        return out

    row = sub.iloc[0]
    if "PL_prg_str" in row.index:
        try:
            out["PL_prg_str"] = float(row["PL_prg_str"])
        except Exception:
            pass
    if "PL_swing_B" in row.index:
        try:
            out["PL_swing_B"] = str(row["PL_swing_B"])
        except Exception:
            pass
    if "PL_gap_B" in row.index:
        try:
            out["PL_gap_B"] = float(row["PL_gap_B"])
        except Exception:
            gap = compute_24_gap(df_24, code)
            if gap is not None:
                out["PL_gap_B"] = gap
    else:
        gap = compute_24_gap(df_24, code)
        if gap is not None:
            out["PL_gap_B"] = gap
    return out
