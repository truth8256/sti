# =============================
# File: charts.py
# =============================
from __future__ import annotations
import re
import pandas as pd
import streamlit as st
import altair as alt
from metrics import compute_24_gap


# =============================
# 기본 유틸
# =============================
def _to_pct_float(v, default=None):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    s = str(v).strip().replace(",", "")
    m = re.match(r"^\s*([+-]?\d+(\.\d+)?)\s*%?\s*$", s)
    if not m:
        return default
    x = float(m.group(1))
    if "%" in s:
        return x
    return x * 100.0 if 0 <= x <= 1 else x


def _to_float(v, default=None):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        s = str(v).replace(",", "").strip()
        return float(s) if s not in ("", "nan", "None") else default
    except Exception:
        return default


def _to_int(v, default=None):
    f = _to_float(v, default=None)
    try:
        return int(f) if f is not None else default
    except Exception:
        return default


def _fmt_pct(x):
    return f"{x:.2f}%" if isinstance(x, (int, float)) else "N/A"


def _fmt_gap(x):
    return f"{x:.2f}p" if isinstance(x, (int, float)) else "N/A"


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df
    out = df.copy()
    out.columns = [str(c).strip().replace("\n", "").replace("\r", "") for c in out.columns]
    return out


# =============================
# 파이차트
# =============================
def _pie_chart(title: str, labels: list[str], values: list[float], colors: list[str],
               width: int = 260, height: int = 260):
    vals = [(v if isinstance(v, (int, float)) and v > 0 else 0.0) for v in values]
    total = sum(vals)
    if total <= 0:
        st.info(f"{title} 자료가 없습니다.")
        return
    vals = [v / total * 100.0 for v in vals]
    df = pd.DataFrame({"구성": labels, "비율": vals})
    chart = (
        alt.Chart(df)
        .mark_arc(innerRadius=60, stroke="white", strokeWidth=1)
        .encode(
            theta=alt.Theta("비율:Q"),
            color=alt.Color("구성:N",
                            scale=alt.Scale(domain=labels, range=colors),
                            legend=None),
            tooltip=[alt.Tooltip("구성:N"), alt.Tooltip("비율:Q", format=".1f")]
        )
        .properties(title=title, width=width, height=height)
    )
    st.altair_chart(chart, use_container_width=False)


# =============================
# 지표
# =============================
def render_results_2024_card(res_row: pd.DataFrame, df_24: pd.DataFrame = None, code: str = None):
    if res_row is None or res_row.empty:
        st.info("해당 선거구의 24년 결과 데이터가 없습니다.")
        return
    res_row = _norm_cols(res_row)
    if "연도" in res_row.columns:
        try:
            cands = res_row.dropna(subset=["연도"]).copy()
            cands["__year__"] = pd.to_numeric(cands["연도"], errors="coerce")
            if (cands["__year__"] == 2024).any():
                r = cands[cands["__year__"] == 2024].iloc[0]
            else:
                r = cands.loc[cands["__year__"].idxmax()]
        except Exception:
            r = res_row.iloc[0]
    else:
        r = res_row.iloc[0]

    c1n = next((c for c in ["후보1_이름", "1위이름", "1위 후보"] if c in res_row.columns), None)
    c1v = next((c for c in ["후보1_득표율", "1위득표율"] if c in res_row.columns), None)
    c2n = next((c for c in ["후보2_이름", "2위이름", "2위 후보"] if c in res_row.columns), None)
    c2v = next((c for c in ["후보2_득표율", "2위득표율"] if c in res_row.columns), None)

    name1 = str(r.get(c1n)) if c1n else "1위"
    share1 = _to_pct_float(r.get(c1v))
    name2 = str(r.get(c2n)) if c2n else "2위"
    share2 = _to_pct_float(r.get(c2v))
    gap = None
    if isinstance(share1, (int, float)) and isinstance(share2, (int, float)):
        gap = round(share1 - share2, 2)
    elif df_24 is not None and code is not None:
        gap = compute_24_gap(df_24, code)

    with st.container(border=True):
        st.markdown("**24년 총선결과**")
        c1, c2, c3 = st.columns(3)
        c1.metric(label=name1, value=_fmt_pct(share1))
        c2.metric(label=name2, value=_fmt_pct(share2))
        c3.metric(label="1~2위 격차", value=_fmt_gap(gap))


def render_incumbent_card(cur_row: pd.DataFrame):
    if cur_row is None or cur_row.empty:
        st.info("현직 정보 데이터가 없습니다.")
        return
    cur_row = _norm_cols(cur_row)
    r = cur_row.iloc[0]
    name_col = next((c for c in ["의원명", "이름", "성명"] if c in cur_row.columns), None)
    party_col = next((c for c in ["정당", "소속정당"] if c in cur_row.columns), None)
    term_col = next((c for c in ["선수", "당선횟수"] if c in cur_row.columns), None)
    age_col = next((c for c in ["연령", "나이"] if c in cur_row.columns), None)
    gender_col = next((c for c in ["성별"] if c in cur_row.columns), None)

    with st.container(border=True):
        st.markdown("**현직정보**")
        st.write(f"- 의원: **{r.get(name_col, 'N/A')}** / 정당: **{r.get(party_col, 'N/A')}**")
        st.write(
            f"- 선수: **{r.get(term_col, 'N/A')}** / 성별: **{r.get(gender_col, 'N/A')}** / 연령: **{r.get(age_col, 'N/A')}**"
        )


def render_prg_party_box(prg_row: pd.DataFrame, pop_row: pd.DataFrame):
    with st.container(border=True):
        st.markdown("**진보당 현황**")
        if prg_row is None or prg_row.empty:
            st.info("진보당 관련 데이터가 없습니다.")
            return
        prg_row = _norm_cols(prg_row)
        r = prg_row.iloc[0]
        strength_col = next((c for c in ["진보당 득표력", "득표력"] if c in prg_row.columns), None)
        st.metric("진보득표력", _fmt_pct(_to_pct_float(r.get(strength_col))))


def render_vote_trend_chart(ts: pd.DataFrame):
    if ts is None or ts.empty:
        st.info("득표 추이 데이터가 없습니다.")
        return
    df = _norm_cols(ts)
    if "연도" in df.columns and "민주" in df.columns:
        df = df.melt(id_vars="연도", var_name="계열", value_name="득표율")
    party_order = ["민주", "보수", "진보", "기타"]
    party_colors = ["#152484", "#E61E2B", "#450693", "#798897"]
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x="연도:O",
            y=alt.Y("득표율:Q", title="득표율(%)"),
            color=alt.Color("계열:N",
                            scale=alt.Scale(domain=party_order, range=party_colors),
                            legend=alt.Legend(orient="top")),
        )
        .properties(height=300)
    )
    with st.container(border=True):
        st.markdown("**정당성향별 득표추이**")
        st.altair_chart(chart, use_container_width=True)


def render_population_box(pop_df: pd.DataFrame):
    with st.container(border=True):
        st.markdown("**인구 정보**")
        st.info("유권자 이동, 연령 구성, 성비 차트 자리")


# =============================
# 레이아웃
# =============================
def render_region_detail_layout(
    df_pop: pd.DataFrame | None = None,
    df_trend: pd.DataFrame | None = None,
    df_24: pd.DataFrame | None = None,
    df_cur: pd.DataFrame | None = None,
    df_prg: pd.DataFrame | None = None,
):
    """
    지역별 페이지 전체 구조 틀
    - 상단: 인구 정보 (1:1 비율)
        - 왼쪽(1): 내부 1:2 비율 → 유권자 이동 / (연령 구성 + 성비)
    - 중간: 정당성향별 득표추이 (단독)
    - 하단: 24년 총선결과 / 현직 정보 / 진보당 현황 (1:1:1)
    """
    import streamlit as st  # 안전차원 (상단 import가 이미 있으면 제거해도 됨)

    # ============ 상단 인구정보 ============ #
    st.markdown("### 👥 인구 정보")
    top_left, top_right = st.columns(2)

    # 왼쪽: 다시 1:2로 세분
    left_small, left_large = top_left.columns([1, 2])

    with left_small.container(border=True, height="stretch"):
        st.markdown("#### 유권자 이동")
        st.info("세로 막대차트 (예: 인구 이동률) 준비중")

    with left_large:
        subcol1, subcol2 = st.columns(2)
        with subcol1.container(border=True, height="stretch"):
            st.markdown("#### 연령 구성")
            st.info("파이차트 (예: 청년층/중년층/고령층 비율) 준비중")
        with subcol2.container(border=True, height="stretch"):
            st.markdown("#### 성비")
            st.info("가로 막대차트 (남/여 비율) 준비중")

    with top_right.container(border=True, height="stretch"):
        st.markdown("#### (추가 정보 공간)")
        st.caption("추후 필요 시 우측 패널에 다른 지표 배치 가능")

    # ============ 중간: 득표 추이 ============ #
    st.markdown("### 📈 정당성향별 득표추이")
    with st.container(border=True):
        st.info("꺾은선그래프 자리 (정당별 연도별 득표율)")

    # ============ 하단: 24년 결과 / 현직 / 진보당 ============ #
    st.markdown("### 🗳️ 선거 결과 및 정치지형")
    col1, col2, col3 = st.columns(3)
    with col1.container(border=True):
        st.markdown("#### 24년 총선결과")
        st.info("총선 결과 카드 자리")
    with col2.container(border=True):
        st.markdown("#### 현직 정보")
        st.info("현직 의원 정보 카드 자리")
    with col3.container(border=True):
        st.markdown("#### 진보당 현황")
        st.info("진보당 현황 카드 자리")




