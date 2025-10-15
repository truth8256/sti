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
# ---- 색상 매핑 유틸 (전역 정의만) ----
def _party_chip_color(name: str) -> tuple[str, str]:
    s = (name or "").strip()
    MAP = [
        ("더불어민주당", ("#152484", "rgba(21,36,132,0.08)")),
        ("국민의힘",     ("#E61E2B", "rgba(230,30,43,0.10)")),
        ("개혁신당",     ("#798897", "rgba(121,136,151,0.12)")),
    ]
    for key, col in MAP:
        if key in s:
            return col
    return ("#334155", "rgba(51,65,85,0.08)")  # default

# ---- 24년 결과 카드 ----
def render_results_2024_card(res_row: pd.DataFrame, df_24: pd.DataFrame = None, code: str = None):
    import re
    from streamlit.components.v1 import html as html_component

    if res_row is None or res_row.empty:
        st.info("해당 선거구의 24년 결과 데이터가 없습니다.")
        return

    res_row = _norm_cols(res_row)

    # 2024 우선, 없으면 최신 연도
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

    # 후보{n}_이름 / 후보{n}_득표율 → 상위 2명
    name_cols = [c for c in res_row.columns if re.match(r"^후보\d+_이름$", c)]
    def share_col_for(n: str) -> str | None:
        for cand in (f"후보{n}_득표율", f"후보{n}_득표율(%)"):
            if cand in res_row.columns:
                return cand
        return None
    pairs = []
    for nc in name_cols:
        n = re.findall(r"\d+", nc)[0]
        sc = share_col_for(n)
        if not sc: continue
        nm = str(r.get(nc)) if pd.notna(r.get(nc)) else None
        sh = _to_pct_float(r.get(sc))
        if nm and isinstance(sh, (int, float)):
            pairs.append((nm, sh))

    if pairs:
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
        top2 = pairs_sorted[:2] if len(pairs_sorted) >= 2 else [pairs_sorted[0], ("2위", None)]
    else:
        # 구형 스키마 fallback
        c1n = next((c for c in ["후보1_이름","1위이름","1위 후보"] if c in res_row.columns), None)
        c1v = next((c for c in ["후보1_득표율","1위득표율","1위득표율(%)"] if c in res_row.columns), None)
        c2n = next((c for c in ["후보2_이름","2위이름","2위 후보"] if c in res_row.columns), None)
        c2v = next((c for c in ["후보2_득표율","2위득표율","2위득표율(%)"] if c in res_row.columns), None)
        name1 = str(r.get(c1n)) if c1n else "1위"; share1 = _to_pct_float(r.get(c1v))
        name2 = str(r.get(c2n)) if c2n else "2위"; share2 = _to_pct_float(r.get(c2v))
        top2 = [(name1, share1), (name2, share2)]

    name1, share1 = top2[0][0] or "1위", top2[0][1]
    name2, share2 = (top2[1][0] or "2위", top2[1][1]) if len(top2) > 1 else ("2위", None)
    gap = round(share1 - share2, 2) if isinstance(share1,(int,float)) and isinstance(share2,(int,float)) \
          else (compute_24_gap(df_24, code) if (df_24 is not None and code is not None) else None)

    # ---- 렌더링(테두리/제목/3열 고정) ----
    with st.container(border=True):
        st.markdown("**24년 총선결과**")
    
        c1_fg, c1_bg = _party_chip_color(name1)
        c2_fg, c2_bg = _party_chip_color(name2)
    
        # 정당명/후보명 분리 (엔터 삽입)
        def split_name(nm: str):
            # “정당 후보명” 구조를 줄바꿈 처리
            parts = nm.strip().split()
            if len(parts) >= 2:
                return parts[0], " ".join(parts[1:])
            return nm, ""
        p1, cand1 = split_name(name1)
        p2, cand2 = split_name(name2)
    
        html = f"""
        <div style="display:grid; grid-template-columns: repeat(3, 1fr); align-items:center; margin-top:6px;">
            <div style="padding:10px 8px; text-align:center;">
                <div style="
                    display:inline-flex; flex-direction:column; align-items:center;
                    padding:6px 10px; border-radius:14px;
                    font-weight:600; font-size:.95rem;
                    color:{c1_fg}; background:{c1_bg}; line-height:1.2;">
                    <span style="opacity:0.9;">{p1}</span>
                    <span style="color:#111827;">{cand1}</span>
                </div>
                <div style="font-weight:700; font-size:1.05rem; margin-top:8px;
                            font-variant-numeric:tabular-nums; letter-spacing:-0.2px; color:#111827;">
                    {_fmt_pct(share1)}
                </div>
            </div>
    
            <div style="padding:10px 8px; text-align:center; border-left:1px solid #EEF2F7;">
                <div style="
                    display:inline-flex; flex-direction:column; align-items:center;
                    padding:6px 10px; border-radius:14px;
                    font-weight:600; font-size:.95rem;
                    color:{c2_fg}; background:{c2_bg}; line-height:1.2;">
                    <span style="opacity:0.9;">{p2}</span>
                    <span style="color:#111827;">{cand2}</span>
                </div>
                <div style="font-weight:700; font-size:1.05rem; margin-top:8px;
                            font-variant-numeric:tabular-nums; letter-spacing:-0.2px; color:#111827;">
                    {_fmt_pct(share2)}
                </div>
            </div>
    
            <div style="padding:10px 8px; text-align:center; border-left:1px solid #EEF2F7;">
                <div style="color:#6B7280; font-weight:600;">1~2위 격차</div>
                <div style="font-weight:700; font-size:1.05rem; margin-top:8px;
                            font-variant-numeric:tabular-nums; letter-spacing:-0.2px; color:#334155;">
                    {_fmt_gap(gap)}
                </div>
            </div>
        </div>
        """
        from streamlit.components.v1 import html as html_component
        html_component(html, height=150, scrolling=False)

# 현직 정보
def render_incumbent_card(cur_row: pd.DataFrame):
    from streamlit.components.v1 import html as html_component

    # 데이터 체크
    if cur_row is None or cur_row.empty:
        with st.container(border=True):
            st.markdown("**현직정보**")
            st.info("현직 정보 데이터가 없습니다.")
        return

    cur_row = _norm_cols(cur_row)
    r = cur_row.iloc[0]

    # 컬럼 자동 탐색
    name_col   = next((c for c in ["의원명","이름","성명"] if c in cur_row.columns), None)
    party_col  = next((c for c in ["정당","소속정당"] if c in cur_row.columns), None)
    term_col   = next((c for c in ["선수","당선횟수"] if c in cur_row.columns), None)
    age_col    = next((c for c in ["연령","나이"] if c in cur_row.columns), None)
    gender_col = next((c for c in ["성별"] if c in cur_row.columns), None)

    name   = str(r.get(name_col, "정보없음")) if name_col else "정보없음"
    party  = str(r.get(party_col, "정당미상")) if party_col else "정당미상"
    term   = str(r.get(term_col, "N/A")) if term_col else "N/A"
    gender = str(r.get(gender_col, "N/A")) if gender_col else "N/A"
    age    = str(r.get(age_col, "N/A")) if age_col else "N/A"

    # 이니셜(아바타용, 없으면 'NA')
    def _initials(s: str) -> str:
        s = (s or "").strip()
        if not s: return "NA"
        # 한글 포함이면 앞 2글자, 아니면 단어 첫글자 2개
        if any('\uac00' <= ch <= '\ud7a3' for ch in s):
            return s[:2]
        parts = [p for p in s.split() if p]
        return (parts[0][:2] if len(parts) == 1 else (parts[0][0] + parts[1][0])).upper()

    ini = _initials(name)

    # 정당 색상
    try:
        fg, bg = _party_chip_color(party)
    except Exception:
        fg, bg = "#334155", "rgba(51,65,85,0.08)"

    # 렌더
    with st.container(border=True):
        st.markdown("**현직정보**")

        html = f"""
        <div style="display:grid; grid-template-columns:72px 1fr; gap:14px; align-items:center; margin-top:6px;">
          <!-- 아바타 -->
          <div style="display:flex; align-items:center; justify-content:center;">
            <div style="width:60px; height:60px; border-radius:50%;
                        background:{bg}; color:{fg};
                        display:flex; align-items:center; justify-content:center;
                        font-weight:700; font-size:1.0rem;">
              {ini}
            </div>
          </div>

          <!-- 본문 -->
          <div>
            <!-- 이름 + 정당칩 -->
            <div style="display:flex; flex-wrap:wrap; align-items:center; gap:8px;">
              <div style="font-size:1.05rem; font-weight:700; color:#111827;">{name}</div>
              <div style="display:inline-flex; align-items:center; gap:6px;
                          padding:4px 10px; border-radius:999px;
                          font-weight:600; font-size:.92rem;
                          color:{fg}; background:{bg};">
                {party}
              </div>
            </div>

            <!-- 배지들 -->
            <div style="display:flex; flex-wrap:wrap; gap:6px; margin-top:10px;">
              <span style="padding:4px 10px; border-radius:999px; background:#F3F4F6; color:#374151; font-size:.88rem; font-weight:600;">선수: {term}</span>
              <span style="padding:4px 10px; border-radius:999px; background:#F3F4F6; color:#374151; font-size:.88rem; font-weight:600;">성별: {gender}</span>
              <span style="padding:4px 10px; border-radius:999px; background:#F3F4F6; color:#374151; font-size:.88rem; font-weight:600;">연령: {age}</span>
            </div>
          </div>
        </div>
        """
        # 높이는 150~180 정도
        html_component(html, height=150, scrolling=False)

# 진보당 현황
def render_prg_party_box(
    prg_row: pd.DataFrame | None,
    pop_row: pd.DataFrame | None = None,
    *,
    code: str | int | None = None,      # 선택된 지역 코드가 있으면 넣어주세요
    region: str | None = None,          # 또는 지역명(선거구명)
    debug: bool = False                 # True면 매칭 과정 캡션으로 보여줌
):
    """index_sample.csv에서 '진보정당 득표력', '진보당 당원수'를
    각각 '진보 득표력', '진보당 당원수'로 깔끔히 표시.
    prg_row가 비어오면 내부에서 index_sample.csv를 직접 읽어 code/region으로 1행을 찾아 씀.
    """
    from streamlit.components.v1 import html as html_component

    def _norm(s: str) -> str:
        s = str(s).replace("\n", " ").replace("\r", " ").strip()
        return " ".join(s.split())

    # 필요시 CSV에서 직접 로드해 1행 선택
    if prg_row is None or prg_row.empty:
        # 1) CSV 로드
        try:
            df_all = pd.read_csv("/mnt/data/index_sample.csv")
        except UnicodeDecodeError:
            df_all = pd.read_csv("/mnt/data/index_sample.csv", encoding="cp949")
        df_all = _norm_cols(df_all)
        df_all.columns = [_norm(c) for c in df_all.columns]

        # 2) 키 컬럼 추정
        code_col = next((c for c in df_all.columns
                         if c in ["code","코드","지역코드","선거구코드","행정구역코드"]), None)
        name_col = next((c for c in df_all.columns
                         if c in ["지역명","선거구명","지역구","지역","구시군"]), None)

        prg_row = pd.DataFrame()

        # 3) 매칭 로직
        if code is not None and code_col:
            # 문자열/숫자 혼합 대비
            def _to_str(x): return _norm(x) if pd.notna(x) else ""
            code_str = _norm(code)
            # 정확 일치 우선
            prg_row = df_all[_to_str(df_all[code_col]) == code_str].head(1)
            # 숫자 일치 보조
            if prg_row.empty:
                try:
                    code_num = pd.to_numeric(code_str, errors="coerce")
                    col_num = pd.to_numeric(df_all[code_col], errors="coerce")
                    prg_row = df_all[col_num == code_num].head(1)
                except Exception:
                    pass

        if (prg_row is None or prg_row.empty) and region and name_col:
            region_norm = _norm(region)
            # 정확 일치
            prg_row = df_all[_norm(df_all[name_col]) == region_norm].head(1)
            # 부분 일치
            if prg_row.empty:
                prg_row = df_all[df_all[name_col].astype(str).str.contains(region_norm, na=False)].head(1)

        # 마지막 안전장치: 그래도 비면 첫 행
        if prg_row is None or prg_row.empty:
            prg_row = df_all.head(1)

        if debug:
            st.caption(f"[debug] index_sample.csv 컬럼: {list(df_all.columns)}")
            st.caption(f"[debug] code_col={code_col!r}, name_col={name_col!r}, 선택행 여부={not prg_row.empty}")

    # --------- 여기서부터 기존 표시 로직 ---------
    with st.container(border=True):
        st.markdown("**진보당 현황**")

        if prg_row is None or prg_row.empty:
            st.info("진보당 관련 데이터가 없습니다. (선택된 행이 비어있음)")
            return

        df = prg_row.copy()
        df.columns = [_norm(c) for c in df.columns]
        r = df.iloc[0]

        # 정확/부분 일치로 두 컬럼 찾기
        def _pick_col(want: str) -> str | None:
            want_n = _norm(want).lower()
            mapping = {_norm(c).lower(): c for c in df.columns}
            if want_n in mapping:
                return mapping[want_n]
            for k, orig in mapping.items():
                if want_n in k:
                    return orig
            return None

        col_strength = _pick_col("진보정당 득표력")
        col_members  = _pick_col("진보당 당원수")

        strength = _to_pct_float(r.get(col_strength)) if col_strength else None
        members  = _to_int(r.get(col_members)) if col_members else None

        if debug:
            st.caption(f"[debug] 매칭: 득표력={col_strength!r}, 당원수={col_members!r}")
            st.caption(f"[debug] 선택행 샘플: {r.to_dict()}")

        # CSS 1회 주입
        if "_css_prg_card_simple" not in st.session_state:
            st.markdown("""
            <style>
              .prg-wrap { display:flex; flex-direction:column; gap:10px; margin-top:6px; }
              .metric-box { display:flex; flex-direction:column; align-items:center; justify-content:center; padding:8px 0; }
              .metric-label { color:#6B7280; font-weight:600; font-size:0.95rem; }
              .metric-value { font-weight:800; font-size:1.15rem; color:#111827; font-variant-numeric:tabular-nums; letter-spacing:-0.2px; }
              .divider { height:1px; background:#E5E7EB; margin:4px 0; width:70%; }
            </style>
            """, unsafe_allow_html=True)
            st.session_state["_css_prg_card_simple"] = True

        html = f"""
        <div class="prg-wrap">
          <div class="metric-box">
            <div class="metric-label">진보 득표력</div>
            <div class="metric-value">{_fmt_pct(strength)}</div>
          </div>
          <div class="divider"></div>
          <div class="metric-box">
            <div class="metric-label">진보당 당원수</div>
            <div class="metric-value">{(f"{members:,}명" if isinstance(members, (int,float)) and members is not None else "N/A")}</div>
          </div>
        </div>
        """
        html_component(html, height=150, scrolling=False)

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

















