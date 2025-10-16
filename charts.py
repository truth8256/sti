# =============================
# File: charts.py (robust, v5-safe)
# =============================
from __future__ import annotations

import re, math
import pandas as pd
import streamlit as st
import altair as alt
from metrics import compute_24_gap  # 존재하지 않으면 함수 내부에서 사용 안 하도록 가드

# -----------------------------
# Altair & Streamlit 기본 세팅
# -----------------------------
# Altair v5에서 Streamlit 기본 테마 간섭으로 인한 warning/scale 충돌 방지
alt.data_transformers.disable_max_rows()  # 대용량 시 자동 샘플링 방지
# st.set_option("deprecation.showPyplotGlobalUse", False)  # (여기서는 pyplot 사용 안함)

# -----------------------------
# 유틸
# -----------------------------
def _to_pct_float(v, default=None):
    """문자 '12.3%' 또는 12.3 또는 0.123 -> 12.3 으로 통일"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    try:
        s = str(v).strip().replace(",", "")
        m = re.match(r"^\s*([+-]?\d+(\.\d+)?)\s*%?\s*$", s)
        if not m:
            return default
        x = float(m.group(1))
        if "%" in s:
            return x
        return x * 100.0 if 0 <= x <= 1 else x
    except Exception:
        return default

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
    """헤더 개행/공백 제거"""
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(c).strip().replace("\n", "").replace("\r", "") for c in out.columns]
    return out

def _load_index_df() -> pd.DataFrame | None:
    """진보당 현황 박스에서 보조로 쓰는 index_sample 로더 (경로 여러 개 시도)"""
    paths = [
        "sti/data/index_sample.csv", "./sti/data/index_sample.csv",
        "data/index_sample.csv", "./data/index_sample.csv",
        "index_sample.csv",
        "/mnt/data/index_sample.csv",
        "/mnt/data/index_sample1012.csv",  # 업로드 파일
    ]
    for path in paths:
        try:
            return _norm_cols(pd.read_csv(path))
        except FileNotFoundError:
            continue
        except UnicodeDecodeError:
            try:
                return _norm_cols(pd.read_csv(path, encoding="cp949"))
            except Exception:
                continue
        except Exception:
            continue
    return None

# -----------------------------
# 스타일 상수 & 전역 CSS
# -----------------------------
ROW_MINH = 260
CARD_HEIGHT = 190

COLOR_TEXT_DARK = "#111827"
COLOR_BLUE = "#1E6BFF"

def _inject_global_css():
    st.markdown(
        f"""
        <style>
          .k-card {{ padding:8px 10px; }}
          .k-eq {{ min-height:{ROW_MINH}px; display:flex; flex-direction:column; justify-content:flex-start; }}
          .k-minh-card {{ min-height:{CARD_HEIGHT}px; }}
          .k-kpi-title {{ color:#6B7280; font-weight:600; font-size:.95rem; }}
          .k-kpi-value {{ font-weight:800; font-size:1.18rem; color:#111827; letter-spacing:-0.2px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# 파티 칩 색상
# -----------------------------
def _party_chip_color(name: str) -> tuple[str, str]:
    s = (name or "").strip()
    mapping = [
        ("더불어민주당", ("#152484", "rgba(21,36,132,.08)")),
        ("국민의힘", ("#E61E2B", "rgba(230,30,43,.10)")),
        ("개혁신당", ("#798897", "rgba(121,136,151,.12)")),
        ("정의당", ("#FFB000", "rgba(255,176,0,.12)")),
        ("진보당", ("#C53030", "rgba(197,48,48,.12)")),
    ]
    for key, col in mapping:
        if key in s:
            return col
    return ("#334155", "rgba(51,65,85,.08)")

# =============================
# 인구 정보 (KPI + 유동비율 막대)
# =============================
def render_population_box(pop_df: pd.DataFrame):
    with st.container(border=True):
        st.markdown("<div class='k-eq'>", unsafe_allow_html=True)

        if pop_df is None or pop_df.empty:
            st.info("유동인구/연령/성비 차트를 위한 데이터가 없습니다.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        df = _norm_cols(pop_df.copy())
        code_col  = next((c for c in ["지역구코드","선거구코드","코드","code","CODE"] if c in df.columns), None)
        total_col = next((c for c in ["전체 유권자","전체유권자","total_voters"] if c in df.columns), None)
        float_col = next((c for c in ["유동인구","유권자 이동","floating","mobility"] if c in df.columns), None)

        if not total_col or not float_col:
            st.error("population.csv에서 '전체 유권자' 또는 '유동인구' 컬럼을 찾지 못했습니다.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        def _to_num(x):
            if pd.isna(x): return 0.0
            if isinstance(x,(int,float)): return float(x)
            try: return float(str(x).replace(",","").strip())
            except: return 0.0

        df[total_col] = df[total_col].apply(_to_num)
        df[float_col] = df[float_col].apply(_to_num)

        if code_col:
            agg = df.groupby(code_col, dropna=False)[[total_col,float_col]].sum(min_count=1).reset_index(drop=True)
            total_voters = float(agg[total_col].sum()); floating_pop = float(agg[float_col].sum())
        else:
            total_voters = float(df[total_col].sum());  floating_pop = float(df[float_col].sum())

        mobility_rate = floating_pop/total_voters if total_voters>0 else float("nan")

        # KPI 카드
        st.markdown(f"""
        <div class="k-card" style="display:flex; flex-direction:column; align-items:center; text-align:center;">
          <div class="k-kpi-title">전체 유권자 수</div>
          <div class="k-kpi-value">{int(round(total_voters)):,}명</div>
          <div style="height:6px;"></div>
          <div class="k-kpi-title">유동인구</div>
          <div class="k-kpi-value">{int(round(floating_pop)):,}명</div>
        </div>
        """, unsafe_allow_html=True)

        # ✅ 레이어(텍스트/룰라인) 완전 제거 → Altair v5 TypeError 원천 봉쇄
        if mobility_rate == mobility_rate:
            bar_df = pd.DataFrame({"항목":["유동비율"], "값":[mobility_rate]})
            x_max = 0.10  # 10%

            chart = (
                alt.Chart(bar_df)
                .mark_bar(color=COLOR_BLUE)
                .encode(
                    y=alt.Y("항목:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
                    x=alt.X(
                        "값:Q",
                        title=None,
                        axis=alt.Axis(format=".0%", values=[0, 0.05, x_max]),
                        scale=alt.Scale(domain=[0, x_max]),
                    ),
                    tooltip=[alt.Tooltip("값:Q", title="유동비율", format=".1%")]
                )
                .properties(height=68, padding={"top": 0, "left": 6, "right": 6, "bottom": 4})
            )
            st.altair_chart(chart, use_container_width=True, theme=None)

        st.markdown("</div>", unsafe_allow_html=True)

# =============================
# 연령 구성 (반원 도넛)
# =============================
def render_age_highlight_chart(pop_df: pd.DataFrame, *, box_height_px: int = 240, width_px: int = 300):
    df = _norm_cols(pop_df.copy()) if pop_df is not None else pd.DataFrame()
    if df is None or df.empty:
        st.info("연령 구성 데이터가 없습니다.")
        return

    Y, M, O = "청년층(18~39세)", "중년층(40~59세)", "고령층(65세 이상)"
    T_CANDS = ["전체 유권자 수", "전체 유권자", "전체유권자", "total_voters"]

    for c in (Y, M, O):
        if c not in df.columns:
            st.info(f"연령대 컬럼이 없습니다: {c}")
            return
    total_col = next((c for c in T_CANDS if c in df.columns), None)
    if total_col is None:
        st.info("'전체 유권자 수' 컬럼이 없습니다.")
        return

    # 숫자화
    for c in [Y, M, O, total_col]:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce",
        ).fillna(0)

    y, m, o = float(df[Y].sum()), float(df[M].sum()), float(df[O].sum())
    tot = float(df[total_col].sum())
    if tot <= 0:
        st.info("전체 유권자 수(분모)가 0입니다.")
        return

    labels, values = [Y, M, O], [y, m, o]
    ratios01 = [v / tot for v in values]
    ratios100 = [r * 100 for r in ratios01]

    # 상단에서 라디오(포커스) 먼저 생성 → 즉시 재렌더 안전
    focus = st.radio("강조", labels, index=0, horizontal=True, label_visibility="collapsed")

    width = max(260, int(width_px))
    height = max(220, int(box_height_px))
    inner_r, outer_r = 68, 106
    cx = width / 2
    cy = height * 0.48

    df_vis = pd.DataFrame({"연령": labels, "명": values, "비율": ratios01, "표시비율": ratios100})

    base = (
        alt.Chart(df_vis)
        .properties(width=width, height=height, padding={"top": 0, "left": 0, "right": 0, "bottom": 0})
    )
    theta = alt.Theta("비율:Q", stack=True, scale=alt.Scale(range=[-math.pi / 2, math.pi / 2]))

    arcs = (
        base.mark_arc(innerRadius=inner_r, outerRadius=outer_r, cornerRadius=6, stroke="white", strokeWidth=1)
        .encode(
            theta=theta,
            color=alt.condition(alt.datum.연령 == focus, alt.value(COLOR_BLUE), alt.value("#E5E7EB")),
            tooltip=[
                alt.Tooltip("연령:N", title="연령대"),
                alt.Tooltip("명:Q", title="인원", format=",.0f"),
                alt.Tooltip("표시비율:Q", title="비율(%)", format=".1f"),
            ],
        )
    )

    # 중앙 텍스트
    idx = labels.index(focus)
    big = (
        alt.Chart(pd.DataFrame({"_": [0]}))
        .mark_text(fontSize=34, fontWeight="bold", color="#0f172a")
        .encode(x=alt.value(cx), y=alt.value(cy - 2), text=alt.value(f"{df_vis.loc[idx, '표시비율']:.1f}%"))
    )
    small = (
        alt.Chart(pd.DataFrame({"_": [0]}))
        .mark_text(fontSize=12, color="#475569")
        .encode(x=alt.value(cx), y=alt.value(cy + 18), text=alt.value(focus))
    )

    st.altair_chart(arcs + big + small, use_container_width=False, theme=None)

# =============================
# 성비 (연령×성별 가로막대)
# =============================
def render_sex_ratio_bar(pop_df: pd.DataFrame, *, box_height_px: int = 240):
    if pop_df is None or pop_df.empty:
        st.info("성비 데이터를 표시할 수 없습니다. (population.csv 없음)")
        return

    df = _norm_cols(pop_df.copy())
    age_buckets = ["20대", "30대", "40대", "50대", "60대", "70대 이상"]
    expect = [f"{a} 남성" for a in age_buckets] + [f"{a} 여성" for a in age_buckets]
    miss = [c for c in expect if c not in df.columns]
    if miss:
        st.info("성비용 컬럼이 부족합니다: " + ", ".join(miss))
        return

    def _num(x):
        if pd.isna(x):
            return 0.0
        if isinstance(x, (int, float)):
            return float(x)
        try:
            return float(str(x).replace(",", "").strip())
        except Exception:
            return 0.0

    df_num = df[expect].applymap(_num).fillna(0.0)
    sums = df_num.sum(axis=0)
    if float(sums.sum()) <= 0:
        st.info("성비 데이터(연령×성별)가 모두 0입니다.")
        return

    rows = []
    for a in age_buckets:
        m, f = float(sums[f"{a} 남성"]), float(sums[f"{a} 여성"])
        tot = m + f if (m + f) > 0 else 1.0
        rows += [
            {"연령대": a, "성별": "남성", "명": m, "비율": m / tot, "연령대총합": m + f},
            {"연령대": a, "성별": "여성", "명": f, "비율": f / tot, "연령대총합": m + f},
        ]
    tidy = pd.DataFrame(rows)
    label_map = {"20대": "18–29세", "30대": "30대", "40대": "40대", "50대": "50대", "60대": "60대", "70대 이상": "70대 이상"}
    tidy["연령대표시"] = tidy["연령대"].map(label_map)

    n = tidy["연령대표시"].nunique()
    height_px = max(box_height_px, n * 44 + 24)

    base = (
        alt.Chart(tidy)
        .properties(height=height_px, padding={"top": 0, "left": 8, "right": 8, "bottom": 26})
        .encode(
            y=alt.Y("연령대표시:N", sort=[label_map[a] for a in age_buckets], title=None),
            x=alt.X(
                "비율:Q",
                stack="normalize",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%", values=[0, 0.5, 1.0], title="구성비(%)"),
            ),
            color=alt.Color(
                "성별:N",
                scale=alt.Scale(domain=["남성", "여성"], range=["#3B82F6", "#EF4444"]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("연령대표시:N", title="연령대"),
                alt.Tooltip("성별:N", title="성별"),
                alt.Tooltip("연령대총합:Q", title="해당 연령대 인원", format=",.0f"),
                alt.Tooltip("명:Q", title="성별 인원", format=",.0f"),
                alt.Tooltip("비율:Q", title="해당 연령대 내 비중", format=".1%"),
            ],
        )
    )
    bars = base.mark_bar(size=16)
    rule50 = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeWidth=2, opacity=0.65).encode(x="x:Q")
    st.altair_chart(bars + rule50, use_container_width=True, theme=None)

# =============================
# 정당성향별 득표추이 (라인)
# =============================
def render_vote_trend_chart(ts: pd.DataFrame):
    """
    정렬 규칙(요청):
      ... → 2020 총선 비례 → 2022 대선 → 2022 광역 비례 → 2022 광역단체장 → ...
    """
    if ts is None or ts.empty:
        st.info("득표 추이 데이터가 없습니다.")
        return

    df = _norm_cols(ts.copy())

    label_col = next((c for c in ["계열", "성향", "정당성향", "party_label", "label"] if c in df.columns), None)
    value_col = next((c for c in ["득표율", "비율", "share", "ratio", "pct", "prop"] if c in df.columns), None)
    wide_cols = [c for c in ["민주", "보수", "진보", "기타"] if c in df.columns]

    id_col = next((c for c in ["선거명", "election", "분류", "연도", "year"] if c in df.columns), None)
    year_col = next((c for c in ["연도", "year"] if c in df.columns), None)

    if wide_cols:
        if not id_col:
            st.info("선거명을 식별할 컬럼이 필요합니다.")
            return
        long_df = df.melt(id_vars=id_col, value_vars=wide_cols, var_name="계열", value_name="득표율")
        base_e = long_df[id_col].astype(str)
    else:
        if not (label_col and value_col):
            st.info("정당 성향(계열)과 득표율 컬럼이 필요합니다.")
            return
        long_df = df.rename(columns={label_col: "계열", value_col: "득표율"}).copy()
        if id_col:
            base_e = long_df[id_col].astype(str)
        elif year_col:
            base_e = long_df[year_col].astype(str)
        else:
            st.info("선거명을 식별할 컬럼이 필요합니다.")
            return

    # 코드 → 한글 라벨
    def _norm_token(s: str) -> str:
        s = str(s).strip().replace("-", "_").replace(" ", "_").upper()
        return re.sub(r"_+", "_", s)

    CODE = re.compile(r"^(20\d{2})(?:_([SG]))?_(NA|LOC|PRESIDENT)(?:_(PRO|GOV))?$")

    def to_kr(s: str) -> str:
        key = _norm_token(s)
        m = CODE.fullmatch(key)
        if not m:
            return str(s)
        year, _rg, lvl, kind = m.group(1), m.group(2), m.group(3), m.group(4)
        if lvl == "PRESIDENT":
            return f"{year} 대선"
        if lvl == "NA" and kind == "PRO":
            return f"{year} 총선 비례"
        if lvl == "LOC" and kind == "PRO":
            return f"{year} 광역 비례"
        if lvl == "LOC" and kind == "GOV":
            return f"{year} 광역단체장"
        return s

    long_df["선거명_표시"] = base_e.apply(to_kr)
    long_df = long_df.dropna(subset=["선거명_표시", "계열", "득표율"])
    # 숫자화(득표율이 0~1일 수도, 0~100일 수도 → 0~100으로 통일 표시)
    long_df["득표율"] = pd.to_numeric(long_df["득표율"], errors="coerce")
    # 비율이 0~1 범위로 들어오면 100배
    mask_01 = (long_df["득표율"] <= 1.0) & (long_df["득표율"] >= 0)
    if mask_01.any():
        long_df.loc[mask_01, "득표율"] = long_df.loc[mask_01, "득표율"] * 100.0

    # 정렬용 연도·타입
    long_df["연도"] = pd.to_numeric(long_df["선거명_표시"].str.extract(r"^(20\d{2})")[0], errors="coerce")
    long_df["연도"] = long_df["연도"].fillna(-1).astype(int)

    def etype(s: str) -> str:
        if "대선" in s:
            return "대선"
        if "광역 비례" in s:
            return "광역 비례"
        if "광역단체장" in s:
            return "광역단체장"
        if "총선 비례" in s:
            return "총선 비례"
        return "기타"

    long_df["선거타입"] = long_df["선거명_표시"].map(etype)

    type_rank = {"대선": 1, "광역 비례": 2, "광역단체장": 3, "총선 비례": 4, "기타": 9}
    uniq = long_df[["선거명_표시", "연도", "선거타입"]].drop_duplicates().copy()
    uniq["순위"] = uniq["선거타입"].map(type_rank).fillna(9)
    uniq = uniq.sort_values(["연도", "순위", "선거명_표시"])
    order = uniq["선거명_표시"].tolist()

    # 사용자 지정 재배치: 2020 총선 비례 뒤에 2022 대선 → 2022 광역 비례 → 2022 광역단체장
    def _first_label(labels, patt):
        for s in labels:
            if (hasattr(patt, "search") and patt.search(s)) or (isinstance(patt, str) and patt in s):
                return s
        return None

    def reorder_after(base_list, anchor_pat, targets_in_order):
        labels = base_list[:]
        anchor = _first_label(labels, anchor_pat)
        if not anchor:
            return labels
        to_insert = []
        for t in targets_in_order:
            lab = _first_label(labels, t)
            if lab and lab in labels:
                labels.remove(lab)
                to_insert.append(lab)
        idx = labels.index(anchor)
        for t in reversed(to_insert):
            labels.insert(idx + 1, t)
        return labels

    order = reorder_after(
        order,
        re.compile(r"^2020.*총선\s*비례"),
        [re.compile(r"^2022.*대선"), re.compile(r"^2022.*광역\s*비례"), re.compile(r"^2022.*광역단체장")],
    )

    party_order = ["민주", "보수", "진보", "기타"]
    color_map = {"민주": "#152484", "보수": "#E61E2B", "진보": "#7B2CBF", "기타": "#6C757D"}
    present = [p for p in party_order if p in long_df["계열"].unique().tolist()]
    colors = [color_map.get(p, "#6B7280") for p in present]

    # 인터랙션(마우스오버 포인트 강조)
    sel = alt.selection_point(fields=["선거명_표시", "계열"], nearest=True, on="mouseover", empty=False)

    base = alt.Chart(long_df).properties(
        height=340, padding={"top": 0, "left": 8, "right": 8, "bottom": 8}
    )

    line = base.mark_line(point=False, strokeWidth=3).encode(
        x=alt.X(
            "선거명_표시:N",
            scale=alt.Scale(domain=order),
            axis=alt.Axis(labelAngle=-32, labelOverlap=False, labelPadding=6, labelLimit=280),
            title="선거명",
        ),
        y=alt.Y("득표율:Q", title="득표율(%)", scale=alt.Scale(zero=True)),
        color=alt.Color("계열:N", scale=alt.Scale(domain=present, range=colors), legend=alt.Legend(title=None, orient="top")),
    )

    hit = base.mark_circle(size=600, opacity=0).encode(
        x="선거명_표시:N", y="득표율:Q", color=alt.Color("계열:N", legend=None)
    ).add_params(sel)

    pts = base.mark_circle(size=120).encode(
        x=alt.X("선거명_표시:N", scale=alt.Scale(domain=order)),
        y="득표율:Q",
        color=alt.Color("계열:N", scale=alt.Scale(domain=present, range=colors), legend=None),
        opacity=alt.condition(sel, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip("선거명_표시:N", title="선거명"),
            alt.Tooltip("계열:N", title="계열"),
            alt.Tooltip("득표율:Q", title="득표율(%)", format=".1f"),
        ],
    ).transform_filter(sel)

    # 연도 밴드 (있을 때만)
    years = sorted([y for y in long_df["연도"].unique().tolist() if y > 0])
    bands = []
    for y in years:
        labels = [l for l in order if re.match(fr"^{y}", l)]
        if labels:
            bands.append({"f": labels[0], "t": labels[-1], "연도": y})

    if bands:
        bg = alt.Chart(pd.DataFrame(bands)).mark_rect(opacity=0.06).encode(
            x=alt.X("f:N", scale=alt.Scale(domain=order), title=None), x2="t:N", color=alt.Color("연도:N", legend=None)
        )
        chart = (bg + line + hit + pts).interactive()
    else:
        chart = (line + hit + pts).interactive()

    with st.container(border=True):
        st.altair_chart(chart, use_container_width=True, theme=None)

# =============================
# 24년 총선 결과 카드
# =============================
def render_results_2024_card(res_row_or_df: pd.DataFrame | None, df_24: pd.DataFrame | None = None, code: str | None = None):
    """
    - res_row_or_df: 단일 선거구 행 또는 해당 선거구만 필터된 DF
    - df_24, code: 둘 다 있으면 compute_24_gap 보조 계산 시도
    """
    with st.container(border=True):
        st.markdown("**24년 총선결과**")

        if res_row_or_df is None or res_row_or_df.empty:
            st.info("해당 선거구의 24년 결과 데이터가 없습니다.")
            return

        res_row = _norm_cols(res_row_or_df)
        # 2024년 행 우선
        try:
            if "연도" in res_row.columns:
                c = res_row.dropna(subset=["연도"]).copy()
                c["__y__"] = pd.to_numeric(c["연도"], errors="coerce")
                if (c["__y__"] == 2024).any():
                    r = c[c["__y__"] == 2024].iloc[0]
                else:
                    r = c.loc[c["__y__"].idxmax()]
            else:
                r = res_row.iloc[0]
        except Exception:
            r = res_row.iloc[0]

        # 후보명/득표율 추출
        name_cols = [c for c in res_row.columns if re.match(r"^후보\d+_이름$", c)]

        def share_col(n):
            for cand in (f"후보{n}_득표율", f"후보{n}_득표율(%)"):
                if cand in res_row.columns:
                    return cand
            return None

        pairs = []
        for nc in name_cols:
            n = re.findall(r"\d+", nc)[0]
            sc = share_col(n)
            if not sc:
                continue
            nm = str(r.get(nc)) if pd.notna(r.get(nc)) else None
            sh = _to_pct_float(r.get(sc))
            if nm and isinstance(sh, (int, float)):
                pairs.append((nm, sh))

        if pairs:
            pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
            top2 = pairs[:2] if len(pairs) >= 2 else [pairs[0], ("2위", None)]
        else:
            c1n = next((c for c in ["후보1_이름", "1위이름", "1위 후보"] if c in res_row.columns), None)
            c1v = next((c for c in ["후보1_득표율", "1위득표율", "1위득표율(%)"] if c in res_row.columns), None)
            c2n = next((c for c in ["후보2_이름", "2위이름", "2위 후보"] if c in res_row.columns), None)
            c2v = next((c for c in ["후보2_득표율", "2위득표율", "2위득표율(%)"] if c in res_row.columns), None)
            top2 = [
                (str(r.get(c1n)) if c1n else "1위", _to_pct_float(r.get(c1v))),
                (str(r.get(c2n)) if c2n else "2위", _to_pct_float(r.get(c2v))),
            ]

        name1, share1 = top2[0][0] or "1위", top2[0][1]
        if len(top2) > 1:
            name2, share2 = top2[1][0] or "2위", top2[1][1]
        else:
            name2, share2 = "2위", None

        # gap 계산: 직접 계산 우선, 없다면 compute_24_gap 보조
        if isinstance(share1, (int, float)) and isinstance(share2, (int, float)):
            gap = round(share1 - share2, 2)
        else:
            try:
                gap = compute_24_gap(df_24, code) if (df_24 is not None and code is not None) else None
            except Exception:
                gap = None

        c1_fg, c1_bg = _party_chip_color(name1)
        c2_fg, c2_bg = _party_chip_color(name2)

        def split(nm: str):
            parts = (nm or "").split()
            return (parts[0], " ".join(parts[1:])) if len(parts) >= 2 else (nm, "")

        p1, cand1 = split(name1)
        p2, cand2 = split(name2)

        html = f"""
        <style>
          .grid-24 {{ display:grid; grid-template-columns: repeat(3,1fr); align-items:center; gap:0; margin-top:4px; }}
          @media (max-width: 720px) {{ .grid-24 {{ grid-template-columns: repeat(2,1fr); gap:8px; }} }}
          .chip {{ display:inline-flex; flex-direction:column; align-items:center; padding:6px 10px; border-radius:14px;
                  font-weight:600; font-size:.95rem; line-height:1.2; }}
          .kpi {{ font-weight:700; font-size:1.02rem; margin-top:8px; font-variant-numeric:tabular-nums; color:{COLOR_TEXT_DARK}; }}
          .cell {{ padding:8px 8px; text-align:center; min-height:80px; }}
          .divider {{ border-left:1px solid #EEF2F7; }}
        </style>
        <div class="k-minh-card">
          <div class="grid-24">
            <div class="cell">
              <div class="chip" style="color:{c1_fg}; background:{c1_bg};"><span style="opacity:.9;">{p1}</span><span style="color:{COLOR_TEXT_DARK};">{cand1}</span></div>
              <div class="kpi">{_fmt_pct(share1)}</div>
            </div>
            <div class="cell divider">
              <div class="chip" style="color:{c2_fg}; background:{c2_bg};"><span style="opacity:.9;">{p2}</span><span style="color:{COLOR_TEXT_DARK};">{cand2}</span></div>
              <div class="kpi">{_fmt_pct(share2)}</div>
            </div>
            <div class="cell divider">
              <div style="color:#6B7280; font-weight:600;">1~2위 격차</div>
              <div class="kpi">{_fmt_gap(gap)}</div>
            </div>
          </div>
        </div>
        """
        from streamlit.components.v1 import html as html_component

        html_component(html, height=CARD_HEIGHT, scrolling=False)

# =============================
# 현직 정보 카드
# =============================
def render_incumbent_card(cur_row: pd.DataFrame | None):
    with st.container(border=True):
        st.markdown("**현직정보**")
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

        career_cols = [c for c in ["최근경력", "주요경력", "경력", "이력", "최근 활동"] if c in cur_row.columns]
        raw = None
        for c in career_cols:
            v = str(r.get(c))
            if v and v.lower() not in ("nan", "none"):
                raw = v
                break

        def _split(s: str) -> list[str]:
            if not s:
                return []
            return [p.strip() for p in re.split(r"[;\n•·/]+", s) if p.strip()]

        items = _split(raw)

        name = str(r.get(name_col, "정보없음")) if name_col else "정보없음"
        party = str(r.get(party_col, "정당미상")) if party_col else "정당미상"
        term = str(r.get(term_col, "N/A")) if term_col else "N/A"
        gender = str(r.get(gender_col, "N/A")) if gender_col else "N/A"
        age = str(r.get(age_col, "N/A")) if age_col else "N/A"
        fg, bg = _party_chip_color(party)

        items_html = "".join([f"<li>{p}</li>" for p in items])
        html = f"""
        <div style="display:flex; flex-direction:column; gap:8px; margin-top:4px;">
          <div style="display:flex; flex-wrap:wrap; align-items:center; gap:8px;">
            <div style="font-size:1.02rem; font-weight:700; color:{COLOR_TEXT_DARK};">{name}</div>
            <div style="display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius:999px; font-weight:600; font-size:.92rem; color:{fg}; background:{bg};">
              {party}
            </div>
          </div>
          <ul style="margin:0; padding-left:1.1rem; color:#374151; font-size:.92rem; line-height:1.65;">
            <li>선수: {term}</li><li>성별: {gender}</li><li>연령: {age}</li>
            {"<li>최근 경력</li><ul style='margin:.2rem 0 0 1.1rem;'>"+items_html+"</ul>" if items_html else ""}
          </ul>
        </div>
        """
        from streamlit.components.v1 import html as html_component

        html_component(html, height=CARD_HEIGHT, scrolling=False)

# =============================
# 진보당 현황 박스
# =============================
def render_prg_party_box(
    prg_row: pd.DataFrame | None,
    pop_row: pd.DataFrame | None = None,
    *,
    code: str | int | None = None,
    region: str | None = None,
    debug: bool = False,
):
    with st.container(border=True):
        st.markdown("**진보당 현황**")

        # 우선 prg_row가 없으면 index_sample에서 추출 시도
        if prg_row is None or prg_row.empty:
            df_all = _load_index_df()
            if df_all is None or df_all.empty:
                st.info("지표 소스(index_sample.csv)를 찾을 수 없습니다.")
                return

            def _norm(s: str) -> str:
                return " ".join(str(s).replace("\n", " ").replace("\r", " ").strip().split())

            df_all.columns = [_norm(c) for c in df_all.columns]
            code_col = "code" if "code" in df_all.columns else None
            name_col = "region" if "region" in df_all.columns else None
            prg_row = pd.DataFrame()
            if code is not None and code_col:
                key = _norm(code)
                prg_row = df_all[df_all[code_col].astype(str).map(_norm) == key].head(1)
            if (prg_row is None or prg_row.empty) and region and name_col:
                key = _norm(region)
                prg_row = df_all[df_all[name_col].astype(str).map(_norm) == key].head(1)
                if prg_row.empty:
                    prg_row = df_all[df_all[name_col].astype(str).str.contains(key, na=False)].head(1)
            if prg_row is None or prg_row.empty:
                prg_row = df_all.head(1)

        df = prg_row.copy()
        df.columns = [" ".join(str(c).split()) for c in df.columns]
        r = df.iloc[0]

        # 지표 컬럼 유연 탐색
        def find_col_exact_or_compact(df, prefer_name, compact_key):
            if prefer_name in df.columns:
                return prefer_name
            for c in df.columns:
                if compact_key in str(c).replace(" ", ""):
                    return c
            return None

        col_strength = find_col_exact_or_compact(df, "진보정당 득표력", "진보정당득표력")
        col_members = find_col_exact_or_compact(df, "진보당 당원수", "진보당당원수")

        strength = _to_pct_float(r.get(col_strength)) if col_strength else None
        members = _to_int(r.get(col_members)) if col_members else None

        from streamlit.components.v1 import html as html_component

        # 상단 KPI 2칸
        html_component(
            f"""
            <div class="k-card" style="display:grid; grid-template-columns:1fr 1fr; align-items:center; gap:12px;">
              <div style="text-align:center;"><div class="k-kpi-title">진보 득표력</div><div class="k-kpi-value">{_fmt_pct(strength) if strength is not None else "N/A"}</div></div>
              <div style="text-align:center;"><div class="k-kpi-title">진보당 당원수</div><div class="k-kpi-value">{(f"{members:,}명" if members is not None else "N/A")}</div></div>
            </div>
            """,
            height=86,
            scrolling=False,
        )

        # 막대 게이지 (0~40%)
        if strength is not None:
            s01 = strength / 100.0
            gdf = pd.DataFrame({"항목": ["진보 득표력"], "값": [s01]})

            base = (
                alt.Chart(gdf)
                .encode(y=alt.Y("항목:N", title=None, axis=alt.Axis(labels=False, ticks=False)))
                .properties(height=64, padding={"top": 0, "left": 6, "right": 6, "bottom": 2})
            )
            g = base.mark_bar(color=COLOR_BLUE).encode(
                x=alt.X("값:Q", axis=alt.Axis(title=None, format=".0%"), scale=alt.Scale(domain=[0, 0.40])),
                tooltip=[alt.Tooltip("값:Q", title="진보 득표력", format=".1%")],
            )
            ticks = alt.Chart(pd.DataFrame({"x": [0.10, 0.20, 0.30, 0.40]})).mark_rule(
                opacity=0.35, strokeDash=[2, 2]
            ).encode(x="x:Q")
            txt = base.mark_text(align="left", dx=4).encode(
                x="값:Q", text=alt.Text("값:Q", format=".1%")
            )
            st.altair_chart(g + ticks + txt, use_container_width=True, theme=None)
        else:
            st.info("진보 득표력 지표가 없습니다.")

# =============================
# 지역 상세 레이아웃
# =============================
def render_region_detail_layout(
    df_pop: pd.DataFrame | None = None,
    df_trend: pd.DataFrame | None = None,
    df_24: pd.DataFrame | None = None,
    df_cur: pd.DataFrame | None = None,
    df_prg: pd.DataFrame | None = None,
):
    _inject_global_css()

    # 인구 정보 섹션
    st.markdown("### 👥 인구 정보")
    left, right = st.columns([1, 5])

    with left:
        render_population_box(df_pop)

    with right:
        a, b = st.columns([1.2, 2.8])
        with a.container(border=True):
            st.markdown("**연령 구성**")
            st.markdown("<div class='k-eq'>", unsafe_allow_html=True)
            render_age_highlight_chart(df_pop, box_height_px=240, width_px=300)
            st.markdown("</div>", unsafe_allow_html=True)
        with b.container(border=True):
            st.markdown("**연령별, 성별 인구분포**")
            st.markdown("<div class='k-eq'>", unsafe_allow_html=True)
            render_sex_ratio_bar(df_pop, box_height_px=240)
            st.markdown("</div>", unsafe_allow_html=True)

    # 득표 추이 섹션
    st.markdown("### 📈 정당성향별 득표추이")
    render_vote_trend_chart(df_trend)

    # 결과/정치지형 섹션
    st.markdown("### 🗳️ 선거 결과 및 정치지형")
    c1, c2, c3 = st.columns(3)
    with c1:
        render_results_2024_card(df_24)  # 내부에서 2024년 자동 선택/보정
    with c2:
        render_incumbent_card(df_cur)
    with c3:
        render_prg_party_box(df_prg, df_pop)

