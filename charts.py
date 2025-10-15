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

# 24년 총선결과

def _party_chip_color(name: str) -> tuple[str, str]:
    """이름 문자열에 정당명이 포함돼 있으면 칩 색상 반환 (텍스트색, 배경색)."""
    s = (name or "").strip()
    MAP = [
        ("더불어민주당", ("#152484", "rgba(21, 36, 132, 0.08)")),
        ("국민의힘",     ("#E61E2B", "rgba(230, 30, 43, 0.10)")),
        ("개혁신당",     ("#798897", "rgba(121, 136, 151, 0.12)")),
    ]
    for key, col in MAP:
        if key in s:
            return col
    return ("#334155", "rgba(51,65,85,0.07)")  # default

def render_results_2024_card(res_row: pd.DataFrame, df_24: pd.DataFrame = None, code: str = None):
    if res_row is None or res_row.empty:
        st.info("해당 선거구의 24년 결과 데이터가 없습니다.")
        return

    # 2024년 연도 선택
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

    # 후보별 득표율 스캔
    import re
    cols = list(res_row.columns)
    name_cols = [c for c in cols if re.match(r"^후보\d+_이름$", c)]
    def share_col_for(n: str) -> str | None:
        for cand in [f"후보{n}_득표율", f"후보{n}_득표율(%)"]:
            if cand in res_row.columns:
                return cand
        return None

    pairs = []
    for nc in name_cols:
        n = re.findall(r"\d+", nc)[0]
        sc = share_col_for(n)
        if sc is None:
            continue
        nm = str(r.get(nc)) if pd.notna(r.get(nc)) else None
        sh = _to_pct_float(r.get(sc))
        if nm and isinstance(sh, (int, float)):
            pairs.append((nm, sh))

    # 상위 2명 선별
    top2 = None
    if pairs:
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
        top2 = pairs_sorted[:2] if len(pairs_sorted) >= 2 else [pairs_sorted[0], ("2위", None)]
    if top2 is None:
        c1n = next((c for c in ["후보1_이름", "1위이름", "1위 후보"] if c in res_row.columns), None)
        c1v = next((c for c in ["후보1_득표율", "1위득표율", "1위득표율(%)"] if c in res_row.columns), None)
        c2n = next((c for c in ["후보2_이름", "2위이름", "2위 후보"] if c in res_row.columns), None)
        c2v = next((c for c in ["후보2_득표율", "2위득표율", "2위득표율(%)"] if c in res_row.columns), None)
        name1 = str(r.get(c1n)) if c1n else "1위"; share1 = _to_pct_float(r.get(c1v))
        name2 = str(r.get(c2n)) if c2n else "2위"; share2 = _to_pct_float(r.get(c2v))
        top2 = [(name1, share1), (name2, share2)]

    name1, share1 = top2[0][0] or "1위", top2[0][1]
    name2, share2 = (top2[1][0] or "2위", top2[1][1]) if len(top2) > 1 else ("2위", None)
    gap = round(share1 - share2, 2) if isinstance(share1, (int,float)) and isinstance(share2, (int,float)) \
          else (compute_24_gap(df_24, code) if (df_24 is not None and code is not None) else None)

    # 스타일
    if "_css_res24" not in st.session_state:
        st.markdown("""
        <style>
        .res24-card { border:1px solid #E5E7EB; border-radius:12px; padding:12px 14px; background:#fff; }
        .res24-grid { display:grid; grid-template-columns: 1fr 1fr 1fr; gap:0; align-items:center; }
        .res24-cell { padding:10px 8px; text-align:center; }
        .res24-cell + .res24-cell { border-left:1px solid #EEF2F7; }  /* 세로 구분선 */
        .res24-title { font-weight:700; font-size:1.05rem; margin:0 0 6px 0; }
        .chip { display:inline-flex; align-items:center; gap:6px; padding:4px 8px; border-radius:999px; font-weight:600; font-size:.98rem; }
        .name { font-weight:600; font-size:.98rem; line-height:1.25; }
        .value { font-weight:700; font-size:1.05rem; margin-top:6px;
                 font-variant-numeric: tabular-nums; letter-spacing:-0.2px; color:#111827;}
        .muted { color:#6B7280; font-weight:600; }
        .value-muted { color:#334155; }
        </style>
        """, unsafe_allow_html=True)
        st.session_state["_css_res24"] = True

    # 칩 색상
    c1_fg, c1_bg = _party_chip_color(name1)
    c2_fg, c2_bg = _party_chip_color(name2)

    with st.container(border=False):
        st.markdown("<div class='res24-card'>", unsafe_allow_html=True)
        st.markdown("<div class='res24-title'>24년 총선결과</div>", unsafe_allow_html=True)

        html = f"""
        <div class="res24-grid">
            <div class="res24-cell">
                <div class="chip" style="color:{c1_fg}; background:{c1_bg};">
                    <span class="name">{name1}</span>
                </div>
                <div class="value">{_fmt_pct(share1)}</div>
            </div>
            <div class="res24-cell">
                <div class="chip" style="color:{c2_fg}; background:{c2_bg};">
                    <span class="name">{name2}</span>
                </div>
                <div class="value">{_fmt_pct(share2)}</div>
            </div>
            <div class="res24-cell">
                <div class="muted">1~2위 격차</div>
                <div class="value value-muted">{_fmt_gap(gap)}</div>
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


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
    col1, col2, col3 = st.columns(3)
    with col1:
        render_results_2024_card(df_24)
    with col2:
        render_incumbent_card(df_cur)
    with col3:
        render_prg_party_box(df_prg, df_pop)





