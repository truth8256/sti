# =============================
# File: metrics.py
# =============================
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union

# [Configuration] List of possible column names for the 'code' or identifier.
# NOTE: To support other code column names, add them to this list.
_CODE_CANDIDATES: List[str] = ["코드", "지역구코드", "선거구코드", "지역코드", "code", "CODE"]

def _canon_code(x: Any) -> str:
    """Canonicalizes the code string for matching (strips non-alphanumeric, leading zeros, converts to lower)."""
    # [Logic] Converts any input to string, removes non-alphanumeric chars, removes leading zeros, and converts to lowercase.
    s: str = str(x).strip()
    s = re.sub(r"[^0-9A-Za-z]", "", s)
    s = s.lstrip("0")
    return s.lower()

def _pct_float(v: Any) -> Optional[float]:
    """Converts a value (like '10.5%', '0.1', or '10.5') to a float percentage (0-100 scale)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    
    s: str = str(v).strip().replace(",", "")
    m: Optional[re.Match[str]] = re.match(r"^\s*([+-]?\d+(\.\d+)?)\s*%?\s*$", s)
    
    if not m:
        return None
        
    try:
        x: float = float(m.group(1))
    except ValueError:
        return None
    
    # If '%' is present, return as is (assumed to be 0-100).
    if "%" in s:
        return x
    
    # If no '%', and value is between 0 and 1 (inclusive), scale it to 0-100. Otherwise, return as is.
    # NOTE: This heuristic assumes values <= 1 are often proportions that need to be scaled up.
    return x * 100.0 if 0 <= x <= 1 else x

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes DataFrame column names by stripping whitespace and newlines."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    df = df.copy()
    df.columns = [str(c).strip().replace("\n", "").replace("\r", "") for c in df.columns]
    # [Maintainability] If further column cleaning (e.g., lowercasing all columns) is needed, implement it here.
    return df

def _detect_code_col(df: pd.DataFrame) -> Optional[str]:
    """Detects the most likely code column name based on a predefined list."""
    # Priority 1: Exact match in original columns
    for cand in _CODE_CANDIDATES:
        if cand in df.columns:
            return cand
    
    # Priority 2: Match in normalized (stripped) columns
    cols: List[str] = [str(c).strip().replace("\n", "").replace("\r", "") for c in df.columns]
    for cand in _CODE_CANDIDATES:
        if cand in cols:
            return df.columns[cols.index(cand)]
    return None

def _get_by_code_local(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """Filters the DataFrame by the canonicalized code value in the detected code column."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    df_norm: pd.DataFrame = _normalize_columns(df)
    
    # Prioritize '코드' column if it exists after normalization, otherwise detect.
    code_col: Optional[str] = "코드" if "코드" in df_norm.columns else _detect_code_col(df_norm)
    
    if not code_col:
        # NOTE: If the code column cannot be detected, this function returns an empty DataFrame.
        return pd.DataFrame()
    
    key: str = _canon_code(code)
    
    try:
        # Apply canonicalization to the code column for robust matching.
        sub: pd.DataFrame = df_norm[df_norm[code_col].astype(str).map(_canon_code) == key]
        return sub if not sub.empty else pd.DataFrame()
    except Exception:
        # Handles potential issues during column conversion/mapping.
        return pd.DataFrame()

def _extract_year_from_election(election: Any) -> Optional[int]:
    """Extracts the four-digit year from an election string (e.g., '2024 General Election' -> 2024)."""
    if not isinstance(election, str):
        return None
    m: Optional[re.Match[str]] = re.match(r"^(\d{4})", election.strip())
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return None

# ----------------------------------------------------------------------

def compute_trend_series(df_trend: pd.DataFrame, code: str) -> pd.DataFrame:
    """
    Processes election trend data (vote_trend.csv) for a specific code.
    Converts 'long' format (election/label/prop) to include 'year' and normalizes proportion.
    'Wide' format (year + series columns) is returned as is.
    """
    sub: pd.DataFrame = _get_by_code_local(df_trend, code)
    if sub.empty:
        return pd.DataFrame()
    
    sub = _normalize_columns(sub)

    # Detect 'long' format columns
    election_col: Optional[str] = "election" if "election" in sub.columns else ("연도" if "연도" in sub.columns else None)
    label_col: Optional[str] = next((c for c in ["label", "성향", "정당성향", "party_label"] if c in sub.columns), None)
    value_col: Optional[str] = next((c for c in ["prop", "득표율", "비율", "share", "ratio", "pct"] if c in sub.columns), None)

    # Handle 'long' format
    if label_col and value_col:
        # Normalize value column to percentage (0-100)
        try:
            if sub[value_col].dtype == "O":
                sub.loc[:, value_col] = sub[value_col].apply(_pct_float)
            elif pd.api.types.is_numeric_dtype(sub[value_col]) and (sub[value_col].dropna() <= 1).all():
                sub.loc[:, value_col] = sub[value_col] * 100.0
        except Exception:
            pass
        
        # Determine the year column
        if election_col == "election":
            sub.loc[:, "year"] = sub["election"].apply(_extract_year_from_election)
        elif election_col == "연도":
            sub.loc[:, "year"] = pd.to_numeric(sub["연도"], errors="coerce")
        else:
            sub.loc[:, "year"] = pd.NA

        # Return relevant columns for 'long' format
        cols_to_keep: List[str] = ["year"] + ([election_col] if election_col else []) + ([label_col] if label_col else []) + ([value_col] if value_col else [])
        cols_to_keep = [c for c in cols_to_keep if c in sub.columns] # Ensure columns exist
        return sub[cols_to_keep].dropna(subset=["year"]).reset_index(drop=True)
    
    # Handle 'wide' format (return as is, after ensuring a 'year' column exists)
    if "연도" in sub.columns:
        sub.loc[:, "year"] = pd.to_numeric(sub["연도"], errors="coerce")
    elif "year" not in sub.columns:
        # If no year column can be inferred, return the subset as is
        return sub

    # [Note for future changes] If 'wide' format requires further column selection or processing (e.g., melt), implement it here.
    return sub

def compute_24_gap(df_24: pd.DataFrame, code: str) -> Optional[float]:
    """
    Calculates the vote share gap (in percentage points) between 1st and 2nd place 
    for the latest year, prioritizing 2024 (5_na_dis_results.csv format).
    """
    try:
        sub: pd.DataFrame = _get_by_code_local(df_24, code)
        if sub.empty:
            return None

        row: pd.Series
        
        # Select the latest year, prioritizing 2024
        if "연도" in sub.columns:
            tmp = sub.dropna(subset=["연도"]).copy()
            tmp.loc[:, "__year__"] = pd.to_numeric(tmp["연도"], errors="coerce")
            
            if (tmp["__year__"] == 2024).any():
                row = tmp[tmp["__year__"] == 2024].iloc[0]
            else:
                max_year_index = tmp["__year__"].idxmax()
                if pd.isna(max_year_index):
                    # No valid year found, fallback to first row
                    row = sub.iloc[0]
                else:
                    row = tmp.loc[max_year_index]
        else:
            # No '연도' column, assume the first row is the relevant data point
            row = sub.iloc[0]

        # Detect 1st and 2nd vote share columns
        c1v: Optional[str] = next((c for c in ["후보1_득표율","1위득표율","1위 득표율","1st_share","득표율_1위","1위득표율(%)"] if c in sub.columns), None)
        c2v: Optional[str] = next((c for c in ["후보2_득표율","2위득표율","2위 득표율","2nd_share","득표율_2위","2위득표율(%)"] if c in sub.columns), None)

        if not (c1v and c2v):
            # [Note for future changes] To support different column names, update the column lists above.
            return None

        # Calculate gap
        v1: Optional[float] = _pct_float(row.get(c1v))
        v2: Optional[float] = _pct_float(row.get(c2v))
        
        if v1 is None or v2 is None:
            return None
            
        return round(v1 - v2, 2)
    except Exception:
        return None

def compute_summary_metrics(df_trend: pd.DataFrame, df_24: pd.DataFrame, df_idx: pd.DataFrame, code: str) -> Dict[str, Union[float, str]]:
    """
    Computes summary metrics (PL_prg_str, PL_swing_B, PL_gap_B) using index data (index_sample1012.csv),
    and falls back to compute_24_gap if PL_gap_B is missing or fails to convert.
    """
    out: Dict[str, Union[float, str]] = {"PL_prg_str": np.nan, "PL_swing_B": "N/A", "PL_gap_B": np.nan}

    sub: pd.DataFrame = _get_by_code_local(df_idx, code)
    
    if sub.empty:
        # Fallback to gap calculation if index data is missing
        gap: Optional[float] = compute_24_gap(df_24, code)
        if gap is not None:
            out["PL_gap_B"] = gap
        return out

    row: pd.Series = sub.iloc[0]
    
    # Extract metrics from index data
    for col, default_val in [("PL_prg_str", np.nan), ("PL_swing_B", "N/A"), ("PL_gap_B", np.nan)]:
        if col in row.index:
            try:
                if col == "PL_prg_str" or col == "PL_gap_B":
                    out[col] = float(row[col])
                elif col == "PL_swing_B":
                    out[col] = str(row[col])
            except Exception:
                # If conversion fails, use default or attempt fallback for PL_gap_B
                if col == "PL_gap_B":
                    gap: Optional[float] = compute_24_gap(df_24, code)
                    if gap is not None:
                        out["PL_gap_B"] = gap
                    # Else, out["PL_gap_B"] remains np.nan from initialization
                else:
                    out[col] = default_val
        elif col == "PL_gap_B":
            # If PL_gap_B column is missing entirely, calculate the gap
            gap: Optional[float] = compute_24_gap(df_24, code)
            if gap is not None:
                out["PL_gap_B"] = gap

    # [Note for future changes] To support additional metrics from the index file, add them to the loop above.
    return out
