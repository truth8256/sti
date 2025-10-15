# =============================
# File: app.py
# =============================
from __future__ import annotations

import re
import streamlit as st
import pandas as pd
from pathlib import Path

from data_loader import (
    load_population_agg,     # population.csv (구 단위 합계본)
    load_party_labels,       # party_labels.csv
    load_vote_trend,         # vote_trend.csv
    load_results_2024,       # 5_na_dis_results.csv
    load_current_info,       # current_info.csv
    load_index_sample,       # index_sample1012.csv (선택)
)

from metrics import (
    compute_trend_series,
    compute_summary_metrics,
    compute_24_gap,
)

from charts import (
    render_population_box,
    render_vote_trend_chart,
    render_results_2024_card,
    render_incumbent_card,
    render_prg_party_box,
)

# -----------------------------
# Page Config
# -----------------------------
APP_TITLE = "지역구 선정 1단계 조사 결과"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🗳️",
    layout="wide",
)

# ---------- Sidebar Navigation ----------
st.sidebar.header("메뉴 선택")
menu = st.sidebar.radio(
    "페이지",
    ["종합", "지역별 분석", "데이터 설명"],
    index=0
)

DATA_DIR = Path("data")

# -----------------------------
# 공통 유틸
# -----------------------------
CODE_CANDIDATES = ["코드", "지역구코드", "선거구코드", "지역코드", "code", "CODE"]
NAME_CANDIDATES = ["지역구", "선거구", "선거구명", "지역명", "district", "지역구명", "region", "지역"]
SIDO_CANDIDATES = ["시/도", "시도", "광역", "sido", "province"]

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if len(df) == 0:
        return df
    df2 = df.copy()
    df2.columns = [str(c).strip().replace("\n", "").replace("\r", "") for c in df2.columns]
    return df2

def _detect_col(df: pd.DataFrame, candidates: list) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    cols = [str(c).strip().replace("\n", "").replace("\r", "") for c in df.columns]
    for cand in candidates:
        if cand in cols:
            return df.columns[cols.index(cand)]
    return None

def _canon_code(x: object) -> str:
    """하이픈/공백 제거, 대소문자 무시, 선행 0 제거 → 코드 표준화"""
    s = str(x).strip()
    s = re.sub(r"[^0-9A-Za-z]", "", s)
    s = s.lstrip("0")
    return s.lower()

def ensure_code_col(df: pd.DataFrame) -> pd.DataFrame:
    """여러 이름의 코드 컬럼을 '코드'(str)로 표준화."""
    if df is None:
        return pd.DataFrame()
    if len(df) == 0:
        return df
    df2 = _normalize_columns(df)
    if "코드" not in df2.columns:
        found = _detect_col(df2, CODE_CANDIDATES)
        if found:
            df2 = df2.rename(columns={found: "코드"})
    if "코드" not in df2.columns:
        idx_name = df2.index.name
        if idx_name and idx_name in CODE_CANDIDATES + ["코드"]:
            df2 = df2.reset_index().rename(columns={idx_name: "코드"})
    if "코드" in df2.columns:
        df2["코드"] = df2["코드"].astype(str)
    else:
        df2["__NO_CODE__"] = True
    return df2

def get_by_code(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """코드 컬럼 자동 탐지 + 표준화 비교로 해당 code 행만 반환(없으면 빈 DF)."""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df2 = _normalize_columns(df)
    code_col = "코드" if "코드" in df2.columns else _detect_col(df2, CODE_CANDIDATES)
    if not code_col:
        return pd.DataFrame()
    try:
        key = _canon_code(code)
        sub = df2[df2[code_col].astype(str).map(_canon_code) == key]
        return sub if len(sub) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def _first_nonempty(*dfs: pd.DataFrame) -> pd.DataFrame | None:
    for d in dfs:
        if isinstance(d, pd.DataFrame) and len(d) > 0:
            return d
    return None

def build_regions(primary_df: pd.DataFrame, *fallback_dfs: pd.DataFrame) -> pd.DataFrame:
    """
    사이드바 선택용 지역 목록: 코드 + 라벨(시/도 + 지역구).
    primary_df가 비어있으면 fallback들(df_24, df_trend, df_curr 등)에서 생성.
    """
    base = _first_nonempty(primary_df, *fallback_dfs)
    if base is None:
        return pd.DataFrame(columns=["코드", "라벨"])
    dfp = ensure_code_col(_normalize_columns(base))

    name_col = _detect_col(dfp, NAME_CANDIDATES)
    if not name_col:
        return (
            dfp.loc[:, ["코드"]]
               .assign(라벨=dfp["코드"])
               .drop_duplicates()
               .sort_values("라벨")
               .reset_index(drop=True)
        )

    sido_col = _detect_col(dfp, SIDO_CANDIDATES)

    def _label(row):
        nm = str(row[name_col]).strip()
        if sido_col and sido_col in row.index and pd.notna(row[sido_col]):
            sido = str(row[sido_col]).strip()
            return nm if nm.startswith(sido) else f"{sido} {nm}"
        return nm

    out = (
        dfp.assign(라벨=dfp.apply(_label, axis=1))
           .loc[:, ["코드", "라벨"]]
           .drop_duplicates()
           .sort_values("라벨")
           .reset_index(drop=True)
    )
    return out

# -----------------------------
# 상단 바 렌더링 (지역별 분석에서만 사용)
# -----------------------------
def render_topbar(page_title: str | None):
    """좌: 페이지별 동적 제목 / 우: 앱 제목(오른쪽 상단 고정)."""
    c1, c2 = st.columns([1, 1])
    with c1:
        if page_title:
            st.title(page_title)
        else:
            # 선택 전에는 빈 영역 유지
            st.write("")
    with c2:
        st.markdown(
            f"""
            <div style="text-align:right; font-weight:700; font-size:1.05rem;">
                🗳️ {APP_TITLE}
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------
# Load Data
# -----------------------------
with st.spinner("데이터 불러오는 중..."):
    df_pop   = load_population_agg(DATA_DIR)       # population.csv
    df_party = load_party_labels(DATA_DIR)         # party_labels.csv
    df_trend = load_vote_trend(DATA_DIR)           # vote_trend.csv
    df_24    = load_results_2024(DATA_DIR)         # 5_na_dis_results.csv
    df_curr  = load_current_info(DATA_DIR)         # current_info.csv
    df_idx   = load_index_sample(DATA_DIR)         # index_sample1012.csv (선택)

# 표준화
df_pop   = ensure_code_col(df_pop)
df_party = ensure_code_col(df_party)
df_trend = ensure_code_col(df_trend)
df_24    = ensure_code_col(df_24)
df_curr  = ensure_code_col(df_curr)
df_idx   = ensure_code_col(df_idx)

# -----------------------------
# Page: 종합
# -----------------------------
if menu == "종합":
    # 기존 형태 유지 (상단 큰 타이틀)
    st.title("🗳️ 지역구 선정 1단계 조사 결과")
    st.caption("에스티아이")

    c1, c2, c3 = st.columns(3)
    with c1:
        n_regions = 0
        if "코드" in df_trend.columns and len(df_trend) > 0:
            n_regions = df_trend["코드"].astype(str).map(_canon_code).nunique()
        elif "코드" in df_pop.columns and len(df_pop) > 0:
            n_regions = df_pop["코드"].astype(str).map(_canon_code).nunique()
        st.metric("지역 수", f"{n_regions:,}")
    with c2:
        st.metric("데이터 소스(표) 수", f"{sum([len(x) > 0 for x in [df_pop, df_24, df_curr, df_trend, df_party, df_idx]])}/6")
    with c3:
        st.metric("최근 파일 로드 상태", "OK" if any(len(x) > 0 for x in [df_pop, df_24, df_curr, df_trend]) else "확인 필요")

    st.divider()
    base_for_sido = _first_nonempty(df_pop, df_trend, df_24, df_curr)
    if base_for_sido is not None:
        base_for_sido = _normalize_columns(base_for_sido)
        base_for_sido = ensure_code_col(base_for_sido)
        sido_col = _detect_col(base_for_sido, SIDO_CANDIDATES)
        if sido_col:
            st.subheader("시/도별 지역구 개수")
            vc = (
                base_for_sido[[sido_col, "코드"]]
                .dropna(subset=[sido_col, "코드"])
                .assign(코드=base_for_sido["코드"].astype(str).map(_canon_code))
                .groupby(sido_col)["코드"].nunique()
                .sort_values(ascending=False)
                .rename("지역구수")
                .to_frame()
            )
            st.dataframe(vc)

# -----------------------------
# Page: 지역별 분석
# -----------------------------
elif menu == "지역별 분석":
    regions = build_regions(df_pop, df_trend, df_24, df_curr)
    if regions.empty:
        render_topbar(None)
        st.error("지역 목록을 만들 수 없습니다. (어느 데이터셋에도 '코드' 및 지역명 컬럼이 없음)")
        st.stop()

    # 선택 전: placeholder를 가진 옵션으로 구성
    PLACEHOLDER = "— 지역을 선택하세요 —"
    options = [PLACEHOLDER] + regions["라벨"].tolist()

    st.sidebar.header("지역 선택")
    sel_label = st.sidebar.selectbox("선거구를 선택하세요", options, index=0)

    # 아직 선택 안 됨 → 상단 우측 앱 제목만, 본문에는 안내 문구
    if sel_label == PLACEHOLDER:
        render_topbar(None)
        st.subheader("지역을 선택하세요")
        st.stop()

    # 선택됨 → 코드 매핑
    sel_code = regions.loc[regions["라벨"] == sel_label, "코드"].iloc[0]

    # 상단바: 왼쪽엔 지역명(동적 타이틀), 오른쪽엔 앱 제목
    render_topbar(sel_label)

# 레이아웃
render_region_detail_layout(
    df_pop=pop_sel,
    df_trend=trend_sel,
    df_24=res_sel,
    df_cur=cur_sel,
    df_prg=prg_sel)

# -----------------------------
# Page: 데이터 설명
# -----------------------------
else:
    st.title("🗳️ 지역구 선정 1단계 조사 결과")
    st.caption("에스티아이")

    st.subheader("데이터 파일 설명")
    st.write("- population.csv: 지역구별 인구/유권자 구조 (구 단위 합계본)")
    st.write("- 5_na_dis_results.csv: 2024 총선 지역구별 1·2위 득표 정보")
    st.write("- current_info.csv: 현직 의원 기본 정보")
    st.write("- vote_trend.csv: 선거별 정당 성향 득표 추이")
    st.write("- party_labels.csv: 정당 코드/라벨 등 매핑 정보")
    st.write("- index_sample1012.csv: 외부 지표(PL/EE 등) *선택*")

    with st.expander("각 DataFrame 컬럼 미리보기"):
        def _cols(df, name):
            st.markdown(f"**{name}**")
            if df is None or len(df) == 0:
                st.write("없음/빈 데이터")
            else:
                st.code(", ".join(map(str, df.columns.tolist())))
        _cols(df_pop,   "df_pop (population)")
        _cols(df_24,    "df_24 (results_2024)")
        _cols(df_curr,  "df_curr (current_info)")
        _cols(df_trend, "df_trend (vote_trend)")
        _cols(df_party, "df_party (party_labels)")
        _cols(df_idx,   "df_idx (index_sample1012)")

st.write("")
st.caption("© 2025 전략지역구 조사 · Streamlit 대시보드")
