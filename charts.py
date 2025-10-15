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
# 지표 컴포넌트
# =============================

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
    # wide → long 변환 (예: '연도', '민주','보수','진보','기타')
    if "연도" in df.columns and any(col in df.columns for col in ["민주", "보수", "진보", "기타"]):
        value_cols = [c for c in ["민주", "보수", "진보", "기타"] if c in df.columns]
        df = df.melt(id_vars="연도", value_vars=value_cols, var_name="계열", value_name="득표율")
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
        if pop_df is None or pop_df.empty:
            st.info("유권자 이동, 연령 구성, 성비 차트를 위한 데이터가 없습니다.")
            return
        # 일단은 자리: 다음 단계에서 실제 세로막대/파이/가로막대로 연결
        st.info("유권자 이동, 연령 구성, 성비 차트 자리")


def render_results_2024_card(res_row: pd.DataFrame, df_24: pd.DataFrame = None, code: str = None):
    """
    2024년(or 최신연도) 결과에서 후보{n}_이름 / 후보{n}_득표율 패턴을 전수 스캔해
    실제 득표율 상위 2명을 자동 선별하여 표시.
    - 퍼센트 문자열('45%')/소수(0.45)/숫자(45) 모두 허용(_to_pct_float 사용)
    - 후보1/후보2 고정이 아니라, 후보3이 1등인 경우도 정확히 집계
    - 기존 1위이름/1위득표율 같은 컬럼은 fallback로만 사용
    """
    if res_row is None or res_row.empty:
        st.info("해당 선거구의 24년 결과 데이터가 없습니다.")
        return

    res_row = _norm_cols(res_row)

    # --- 2024 우선, 없으면 최신 연도 행 선택 ---
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

    # --- 후보{n}_이름 / 후보{n}_득표율 패턴 전수 스캔 ---
    import re
    cols = list(res_row.columns)
    name_cols = [c for c in cols if re.match(r"^후보\d+_이름$", c)]
    # 득표율 컬럼 후보: 일반/괄호% 표기 모두 대응
    def share_col_for(n: str) -> str | None:
        for cand in [f"후보{n}_득표율", f"후보{n}_득표율(%)"]:
            if cand in res_row.columns:
                return cand
        return None

    # (이름, 득표율값) 리스트 구성
    pairs = []
    for nc in name_cols:
        n = re.findall(r"\d+", nc)[0]  # 후보 번호
        sc = share_col_for(n)
        if sc is None:
            continue
        nm = str(r.get(nc)) if pd.notna(r.get(nc)) else None
        sh = _to_pct_float(r.get(sc))
        if nm and isinstance(sh, (int, float)):
            pairs.append((nm, sh))

    # --- 상위 2명 선별 ---
    top2 = None
    if pairs:
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
        if len(pairs_sorted) == 1:
            top2 = [pairs_sorted[0], ("2위", None)]
        else:
            top2 = pairs_sorted[:2]

    # --- fallback: 기존 1위/2위 컬럼(데이터가 구형 스키마일 때) ---
    if top2 is None:
        c1n = next((c for c in ["후보1_이름", "1위이름", "1위 후보"] if c in res_row.columns), None)
        c1v = next((c for c in ["후보1_득표율", "1위득표율", "1위득표율(%)"] if c in res_row.columns), None)
        c2n = next((c for c in ["후보2_이름", "2위이름", "2위 후보"] if c in res_row.columns), None)
        c2v = next((c for c in ["후보2_득표율", "2위득표율", "2위득표율(%)"] if c in res_row.columns), None)

        name1 = str(r.get(c1n)) if c1n else "1위"
        share1 = _to_pct_float(r.get(c1v))
        name2 = str(r.get(c2n)) if c2n else "2위"
        share2 = _to_pct_float(r.get(c2v))
        top2 = [(name1, share1), (name2, share2)]

    # --- 격차 계산 (가능하면 직접, 아니면 compute_24_gap) ---
    share1 = top2[0][1]
    share2 = top2[1][1] if len(top2) > 1 else None
    if isinstance(share1, (int, float)) and isinstance(share2, (int, float)):
        gap = round(share1 - share2, 2)
    else:
        gap = compute_24_gap(df_24, code) if (df_24 is not None and code is not None) else None

    # --- 렌더링 (제목과 같은 글씨 크기) ---
    with st.container(border=True):
        st.markdown("**24년 총선결과**")

        # 후보명이 없는 경우 대비
        name1 = top2[0][0] if top2 and top2[0][0] else "1위"
        name2 = top2[1][0] if len(top2) > 1 and top2[1][0] else "2위"

        html = f"""
        <div style='display:flex; justify-content:space-between; margin-top:10px;'>
            <div style='text-align:center; width:32%;'>
                <div style='font-size:1.05rem; font-weight:600;'>{name1}</div>
                <div style='font-size:1.05rem; font-weight:600; color:#2B4162;'>{_fmt_pct(share1)}</div>
            </div>
            <div style='text-align:center; width:32%;'>
                <div style='font-size:1.05rem; font-weight:600;'>{name2}</div>
                <div style='font-size:1.05rem; font-weight:600; color:#2B4162;'>{_fmt_pct(share2)}</div>
            </div>
            <div style='text-align:center; width:32%;'>
                <div style='font-size:1.05rem; font-weight:600;'>1~2위 격차</div>
                <div style='font-size:1.05rem; font-weight:600; color:#2B4162;'>{_fmt_gap(gap)}</div>
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)


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

    # ============ 상단: 인구정보 ============ #
    st.markdown("### 👥 인구 정보")
    top_left, top_right = st.columns(2)

    # 왼쪽: 다시 1:2로 세분 (현재는 자리, 다음 단계에서 df_pop 기반 차트로 대체)
    left_small, left_large = top_left.columns([1, 2])
    with left_small.container(border=True, height="stretch"):
        st.markdown("#### 유권자 이동")
        st.info("세로 막대차트 (예: 전입/전출, 이동률 등) 자리")
    with left_large:
        subcol1, subcol2 = st.columns(2)
        with subcol1.container(border=True, height="stretch"):
            st.markdown("#### 연령 구성")
            st.info("파이차트 자리")
        with subcol2.container(border=True, height="stretch"):
            st.markdown("#### 성비")
            st.info("가로 막대차트 자리")

    # ============ 중간: 득표 추이(실제 차트 호출) ============ #
    st.markdown("### 📈 정당성향별 득표추이")
    render_vote_trend_chart(df_trend)

    # ============ 하단: 24년 결과 / 현직 / 진보당 (실제 컴포넌트 호출) ============ #
    st.markdown("### 🗳️ 선거 결과 및 정치지형")

# CSS로 동일 높이 컨테이너 스타일 적용
st.markdown("""
    <style>
    .equal-card {
        background-color: rgba(240, 242, 246, 0.6);
        border: 1px solid #d9d9d9;
        border-radius: 10px;
        padding: 15px 18px;
        height: 340px;              /* ✅ 동일 높이 (필요시 조정) */
        display: flex;
        flex-direction: column;
        justify-content: space-between;  /* 내용 아래 여백 자연스럽게 분배 */
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .equal-card h3, .equal-card h4, .equal-card strong {
        margin-top: 0;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    with st.container():
        st.markdown("<div class='equal-card'>", unsafe_allow_html=True)
        render_results_2024_card(df_24)
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown("<div class='equal-card'>", unsafe_allow_html=True)
        render_incumbent_card(df_cur)
        st.markdown("</div>", unsafe_allow_html=True)

with col3:
    with st.container():
        st.markdown("<div class='equal-card'>", unsafe_allow_html=True)
        render_prg_party_box(df_prg, df_pop)
        st.markdown("</div>", unsafe_allow_html=True)





