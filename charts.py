# =============================
# charts.py
# =============================
from __future__ import annotations

import re, math
import pandas as pd
import streamlit as st
import altair as alt
from metrics import compute_24_gap  # Safe import: used only if available

# ---------------------------------------------
# [Altair Base Setup]
# - Disable max_rows to avoid down-sampling.
# - No Streamlit theme overrides or CSS hacks.
# ---------------------------------------------
alt.data_transformers.disable_max_rows()

# ---------------------------------------------
# [Utils]
# ---------------------------------------------
def _norm(x) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^\w]+", "", s)
    return s

def _to_pct_float(v, default=None):
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

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(c).strip().replace("\n", "").replace("\r", "") for c in out.columns]
    return out

def _load_index_df() -> pd.DataFrame | None:
    paths = [
        "sti/data/index_sample.csv", "./sti/data/index_sample.csv",
        "data/index_sample.csv", "./data/index_sample.csv",
        "index_sample.csv",
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

# ---------------------------------------------
# [Colors]
# ---------------------------------------------
COLOR_TEXT_DARK = "#111827"
COLOR_BLUE      = "#1E6BFF"   # Accent for highlights

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

# =========================================================
# [Population Box] KPI + Comparison Bar
# NOTE(1): KPI typography unified with "진보당 현황" box (font-weight:800, dark text).
# NOTE(2): Replaced mobility-rate bar -> single comparison bar:
#          "Region total voters" vs "10-region average voters".
#          Falls back to single bar if average cannot be computed.
# NOTE(4): Set chart height via box_height_px to align with other two boxes.
# =========================================================
def render_population_box(pop_df: pd.DataFrame, *, box_height_px: int = 240):
    with st.container(border=True):
        st.markdown("**인구 정보**")

        if pop_df is None or pop_df.empty:
            st.info("유동인구/연령/성비 차트를 위한 데이터가 없습니다.")
            return

        df = _norm_cols(pop_df.copy())
        code_col  = next((c for c in ["지역구코드","선거구코드","코드","code","CODE"] if c in df.columns), None)
        total_col = next((c for c in ["전체 유권자","전체유권자","total_voters"] if c in df.columns), None)
        if not total_col:
            st.error("population.csv에서 '전체 유권자' 컬럼을 찾지 못했습니다.")
            return

        def _to_num(x):
            if pd.isna(x): return 0.0
            if isinstance(x,(int,float)): return float(x)
            try: return float(str(x).replace(",","").strip())
            except: return 0.0

        df[total_col] = df[total_col].apply(_to_num)

        # region_total: sum within current selection
        if code_col:
            grp = df.groupby(code_col, dropna=False)[[total_col]].sum(min_count=1).reset_index()
            region_total = float(grp[total_col].sum()) if len(grp) == 1 else float(grp[total_col].iloc[0])
            n_regions = len(grp)
            avg_total = float(grp[total_col].mean()) if n_regions >= 2 else None
        else:
            region_total = float(df[total_col].sum())
            avg_total = None  # cannot compute reliably

        # --- KPI (typography unified with "진보당 현황") ---
        st.markdown(
            f"""
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px; margin-top:4px;">
              <div style="text-align:center;">
                <div style="color:#6B7280; font-weight:600; margin-bottom:6px;">전체 유권자 수</div>
                <div style="font-weight:800; color:{COLOR_TEXT_DARK};">{int(round(region_total)):,}명</div>
              </div>
              <div style="text-align:center;">
                <div style="color:#6B7280; font-weight:600; margin-bottom:6px;">10개 지역 평균</div>
                <div style="font-weight:800; color:{COLOR_TEXT_DARK};">{(int(round(avg_total)) if isinstance(avg_total,(int,float)) else 'N/A') if avg_total else 'N/A'}{'' if avg_total else ''}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Comparison bar (region vs average) ---
        if avg_total and isinstance(avg_total, (int, float, float)) and avg_total > 0:
            bar_df = pd.DataFrame({
                "항목": ["해당 지역", "10개 평균"],
                "값": [region_total, avg_total],
            })
            x_max = max(region_total, avg_total) * 1.1
        else:
            bar_df = pd.DataFrame({
                "항목": ["해당 지역"],
                "값": [region_total],
            })
            x_max = region_total * 1.1 if region_total > 0 else 1.0

        chart = (
            alt.Chart(bar_df)
            .mark_bar()
            .encode(
                y=alt.Y("항목:N", title=None, axis=alt.Axis(labels=True, ticks=False)),
                x=alt.X(
                    "값:Q",
                    title=None,
                    axis=alt.Axis(format="~,", labelOverlap=False),
                    scale=alt.Scale(domain=[0, x_max]),
                ),
                tooltip=[alt.Tooltip("항목:N", title="구분"),
                         alt.Tooltip("값:Q", title="유권자수", format=",.0f")],
                color=alt.condition(
                    alt.datum.항목 == "해당 지역",
                    alt.value(COLOR_BLUE),
                    alt.value("#9CA3AF"),
                ),
            )
        ).properties(height=box_height_px)
        st.altair_chart(chart, use_container_width=True, theme=None)

# =========================================================
# [Age Composition: Half Donut]
# NOTE(2): Fixed order; center value shows selected cohort.
# NOTE(4): height coordinated via box_height_px.
# =========================================================
def render_age_highlight_chart(pop_df: pd.DataFrame, *, box_height_px: int = 240, width_px: int = 300):
    df = _norm_cols(pop_df.copy()) if pop_df is not None else pd.DataFrame()
    if df is None or df.empty:
        st.info("연령 구성 데이터가 없습니다."); return

    Y, M, O = "청년층(18~39세)", "중년층(40~59세)", "고령층(65세 이상)"
    T_CANDS = ["전체 유권자 수","전체 유권자","전체유권자","total_voters"]

    for c in (Y, M, O):
        if c not in df.columns:
            st.info(f"연령대 컬럼이 없습니다: {c}"); return
    total_col = next((c for c in T_CANDS if c in df.columns), None)
    if total_col is None:
        st.info("'전체 유권자 수' 컬럼이 없습니다."); return

    for c in [Y, M, O, total_col]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce").fillna(0)

    y, m, o = float(df[Y].sum()), float(df[M].sum()), float(df[O].sum())
    tot = float(df[total_col].sum())
    if tot <= 0: st.info("전체 유권자 수(분모)가 0입니다."); return

    mid_60_64 = max(0.0, tot - (y + m + o))  # remainder = 60–64
    labels_order = [Y, M, "60–64세", O]
    values = [y, m, mid_60_64, o]

    ratios01  = [v/tot for v in values]
    ratios100 = [r*100 for r in ratios01]

    focus = st.radio("강조", [Y, M, O], index=0, horizontal=True, label_visibility="collapsed")

    inner_r, outer_r = 68, 106
    df_vis = pd.DataFrame({
        "연령": labels_order,
        "명": values,
        "비율": ratios01,
        "표시비율": ratios100,
        "강조": [l == focus for l in labels_order],
        "순서": [1, 2, 3, 4],
    })

    chart = (
        alt.Chart(df_vis)
        .mark_arc(innerRadius=inner_r, outerRadius=outer_r, cornerRadius=6, stroke="white", strokeWidth=1)
        .encode(
            theta=alt.Theta("비율:Q", stack=True, sort=None, scale=alt.Scale(range=[-math.pi/2, math.pi/2])),
            order=alt.Order("순서:Q"),
            color=alt.Color("강조:N", scale=alt.Scale(domain=[True, False], range=[COLOR_BLUE, "#E5E7EB"]), legend=None),
            tooltip=[alt.Tooltip("연령:N", title="연령대"),
                     alt.Tooltip("명:Q", title="인원", format=",.0f"),
                     alt.Tooltip("표시비율:Q", title="비율(%)", format=".1f")],
        )
        .properties(width=max(260, int(width_px)), height=max(220, int(box_height_px)))
        .configure_view(stroke=None)
    )
    st.altair_chart(chart, use_container_width=False, theme=None)

    idx = labels_order.index(focus)
    pct_txt = f"{ratios100[idx]:.1f}%"
    st.markdown(
        f"""
        <div style="text-align:center; margin-top:-{int(outer_r*0.85)}px;">
          <div style="font-size:34px; font-weight:800; color:#0f172a; line-height:1;">{pct_txt}</div>
          <div style="font-size:12px; color:#475569; margin-top:2px;">{focus}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# [Sex Composition by Age: Horizontal Bars]
# NOTE(3): x-axis max set to 30% to avoid excessive right padding.
# NOTE(4): height coordinated via box_height_px.
# =========================================================
def render_sex_ratio_bar(pop_df: pd.DataFrame, *, box_height_px: int = 240):
    if pop_df is None or pop_df.empty:
        st.info("성비 데이터를 표시할 수 없습니다. (population.csv 없음)")
        return

    df = _norm_cols(pop_df.copy())
    age_buckets = ["20대","30대","40대","50대","60대","70대 이상"]
    expect = [f"{a} 남성" for a in age_buckets] + [f"{a} 여성" for a in age_buckets]
    miss = [c for c in expect if c not in df.columns]
    if miss:
        st.info("성비용 컬럼이 부족합니다: " + ", ".join(miss))
        return

    def _num(x):
        if pd.isna(x): return 0.0
        if isinstance(x,(int,float)): return float(x)
        try: return float(str(x).replace(",","").strip())
        except: return 0.0

    df_num = df[expect].applymap(_num).fillna(0.0)
    sums = df_num.sum(axis=0)
    grand_total = float(sums.sum())
    if grand_total <= 0:
        st.info("성비 데이터(연령×성별)가 모두 0입니다.")
        return

    rows = []
    for a in age_buckets:
        m, f = float(sums[f"{a} 남성"]), float(sums[f"{a} 여성"])
        rows += [{"연령대":a,"성별":"남성","명":m}, {"연령대":a,"성별":"여성","명":f}]
    tidy = pd.DataFrame(rows)
    tidy["전체비중"] = tidy["명"] / grand_total
    age_tot = tidy.groupby("연령대")["명"].transform("sum").replace(0, 1.0)
    tidy["연령대내비중"] = tidy["명"] / age_tot

    label_map = {"20대":"18–29세","30대":"30대","40대":"40대","50대":"50대","60대":"60대","70대 이상":"70대 이상"}
    tidy["연령대표시"] = tidy["연령대"].map(label_map)

    male_color = "#1E40AF"
    female_color = "#60A5FA"

    bars = (
        alt.Chart(tidy)
        .mark_bar(size=19)
        .encode(
            y=alt.Y("연령대표시:N", sort=[label_map[a] for a in age_buckets], title=None),
            x=alt.X(
                "전체비중:Q",
                scale=alt.Scale(domain=[0, 0.30]),  # <-- NOTE(3): max to 30%
                axis=alt.Axis(format=".0%", title="전체 기준 구성비(%)", grid=True)
            ),
            color=alt.Color(
                "성별:N",
                scale=alt.Scale(domain=["남성","여성"], range=[male_color, female_color]),
                legend=alt.Legend(title=None, orient="top")
            ),
            tooltip=[
                alt.Tooltip("연령대표시:N", title="연령대"),
                alt.Tooltip("성별:N", title="성별"),
                alt.Tooltip("명:Q", title="인원", format=",.0f"),
                alt.Tooltip("전체비중:Q", title="전체 기준 비중", format=".1%"),
                alt.Tooltip("연령대내비중:Q", title="연령대 내부 비중", format=".1%"),
            ],
        )
        .properties(height=box_height_px)
        .configure_view(stroke=None)
    )
    st.altair_chart(bars, use_container_width=True, theme=None)

# =========================================================
# [Vote Trend by Ideology: Line Chart]
# NOTE(5): Increased default height.
# NOTE(6): Legend reserved inside container at top-left (legend orient).
#          Invisible points layer ensures legend always renders.
# =========================================================
def render_vote_trend_chart(ts: pd.DataFrame, *, box_height_px: int = 420):
    import re
    if ts is None or ts.empty:
        st.info("득표 추이 데이터가 없습니다."); return
    df = _norm_cols(ts.copy())

    label_col = next((c for c in ["계열","성향","정당성향","party_label","label"] if c in df.columns), None)
    value_col = next((c for c in ["득표율","비율","share","ratio","pct","prop"] if c in df.columns), None)
    wide_cols  = [c for c in ["민주","보수","진보","기타"] if c in df.columns]

    id_col  = next((c for c in ["선거명","election","분류","연도","year"] if c in df.columns), None)
    year_col = next((c for c in ["연도","year"] if c in df.columns), None)

    if wide_cols:
        if not id_col: st.warning("선거명을 식별할 컬럼이 필요합니다."); return
        long_df = df.melt(id_vars=id_col, value_vars=wide_cols, var_name="계열", value_name="득표율")
        base_e = long_df[id_col].astype(str)
    else:
        if not (label_col and value_col): st.warning("정당 성향(계열)과 득표율 컬럼이 필요합니다."); return
        long_df = df.rename(columns={label_col:"계열", value_col:"득표율"}).copy()
        if id_col: base_e = long_df[id_col].astype(str)
        elif year_col: base_e = long_df[year_col].astype(str)
        else: st.warning("선거명을 식별할 컬럼이 필요합니다."); return

    def _norm_token(s: str) -> str:
        s = str(s).strip().replace("-","_").replace(" ","_").upper()
        return re.sub(r"_+","_", s)
    CODE = re.compile(r"^(20\d{2})(?:_([SG]))?_(NA|LOC|PRESIDENT)(?:_(PRO|GOV))?$")
    def to_kr(s: str) -> str:
        key = _norm_token(s)
        m = CODE.fullmatch(key)
        if not m: return str(s)
        year, _rg, lvl, kind = m.group(1), m.group(2), m.group(3), m.group(4)
        if lvl=="PRESIDENT": return f"{year} 대선"
        if lvl=="NA"  and kind=="PRO": return f"{year} 총선 비례"
        if lvl=="LOC" and kind=="PRO": return f"{year} 광역 비례"
        if lvl=="LOC" and kind=="GOV": return f"{year} 광역단체장"
        return s

    long_df["선거명_표시"] = base_e.apply(to_kr)
    long_df = long_df.dropna(subset=["선거명_표시","계열","득표율"])
    long_df["연도"] = pd.to_numeric(long_df["선거명_표시"].str.extract(r"^(20\d{2})")[0], errors="coerce")

    def etype(s: str) -> str:
        if "대선" in s: return "대선"
        if "광역 비례" in s: return "광역 비례"
        if "광역단체장" in s: return "광역단체장"
        if "총선 비례" in s: return "총선 비례"
        return "기타"
    long_df["선거타입"] = long_df["선거명_표시"].astype(str).map(etype)

    type_rank = {"대선":1, "광역 비례":2, "광역단체장":3, "총선 비례":4, "기타":9}
    uniq = long_df[["선거명_표시","연도","선거타입"]].drop_duplicates().copy()
    uniq["순위"] = uniq["선거타입"].map(type_rank)
    uniq = uniq.sort_values(["연도","순위","선거명_표시"])
    order = uniq["선거명_표시"].dropna().astype(str).tolist()
    if not order:
        order = long_df.sort_values(["연도","선거타입","선거명_표시"])["선거명_표시"].astype(str).unique().tolist()

    long_df["선거명_표시"] = pd.Categorical(long_df["선거명_표시"], categories=order, ordered=True)
    long_df = long_df.sort_values(["선거명_표시", "계열"]).reset_index(drop=True)

    party_order = ["민주","보수","진보","기타"]
    color_map   = {"민주":"#152484", "보수":"#E61E2B", "진보":"#7B2CBF", "기타":"#6C757D"}
    colors      = [color_map[p] for p in party_order]

    # Legend-forcing ghost layer (legend at top-left)
    first_x = order[0] if len(order) else (str(long_df["선거명_표시"].iloc[0]) if not long_df.empty else "기준")
    legend_df = pd.DataFrame({"선거명_표시":[first_x]*len(party_order), "계열":party_order, "득표율":[None]*len(party_order)})
    legend = (
        alt.Chart(legend_df)
        .mark_point(opacity=0.001)
        .encode(
            x=alt.X("선거명_표시:N", sort=None, scale=alt.Scale(domain=order), title=None),
            y=alt.Y("득표율:Q"),
            color=alt.Color("계열:N",
                            scale=alt.Scale(domain=party_order, range=colors),
                            legend=alt.Legend(title=None, orient="top-left", values=party_order))  # NOTE(6)
        )
    )

    line = alt.Chart(long_df).mark_line(point=False, strokeWidth=3).encode(
        x=alt.X("선거명_표시:N", sort=None, scale=alt.Scale(domain=order),
                axis=alt.Axis(labelAngle=-32, labelOverlap=False, labelPadding=6, labelLimit=280), title="선거명"),
        y=alt.Y("득표율:Q", title="득표율(%)"),
        color=alt.Color("계열:N", scale=alt.Scale(domain=party_order, range=colors), legend=None)
    )

    sel = alt.selection_point(fields=["선거명_표시","계열"], nearest=True, on="mouseover", empty=False)
    hit = alt.Chart(long_df).mark_circle(size=600, opacity=0).encode(
        x=alt.X("선거명_표시:N", sort=None),
        y="득표율:Q",
        color=alt.Color("계열:N", scale=alt.Scale(domain=party_order, range=colors), legend=None)
    ).add_params(sel)

    pts = alt.Chart(long_df).mark_circle(size=120).encode(
        x=alt.X("선거명_표시:N", sort=None, scale=alt.Scale(domain=order)),
        y="득표율:Q",
        color=alt.Color("계열:N", scale=alt.Scale(domain=party_order, range=colors), legend=None),
        opacity=alt.condition(sel, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("선거명_표시:N", title="선거명"),
                 alt.Tooltip("계열:N", title="계열"),
                 alt.Tooltip("득표율:Q", title="득표율(%)", format=".1f")]
    ).transform_filter(sel)

    chart = (legend + line + hit + pts).properties(
        height=box_height_px,
        padding={"top": 50, "left": 8, "right": 8, "bottom": 8},  # NOTE(6): top padding space for legend
    ).interactive()

    with st.container(border=True):
        st.altair_chart(chart, use_container_width=True, theme=None)

# =========================================================
# [2024 Results Card]
# =========================================================
def render_results_2024_card(res_row_or_df: pd.DataFrame | None, df_24: pd.DataFrame | None = None, code: str | None = None):
    with st.container(border=True):
        st.markdown("**24년 총선결과**")

        if res_row_or_df is None or res_row_or_df.empty:
            st.info("해당 선거구의 24년 결과 데이터가 없습니다.")
            return

        res_row = _norm_cols(res_row_or_df)
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

        name_cols = [c for c in res_row.columns if re.match(r"^후보\d+_이름$", c)]
        def share_col(n):
            for cand in (f"후보{n}_득표율", f"후보{n}_득표율(%)"):
                if cand in res_row.columns: return cand
            return None

        pairs = []
        for nc in name_cols:
            n = re.findall(r"\d+", nc)[0]
            sc = share_col(n)
            if not sc: continue
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

        gap_txt = f"{gap:.1f} %p" if isinstance(gap,(int,float)) else "N/A"

        html = f"""
        <div style="display:grid; grid-template-columns: 1fr 1fr; align-items:center; gap:0;">
          <div style="text-align:center; padding:8px;">
            <div style="display:inline-flex; padding:6px 10px; border-radius:14px; font-weight:600; color:{c1_fg}; background:{c1_bg};">{p1}</div>
            <div style="font-weight:700; margin-top:6px; color:{COLOR_TEXT_DARK};">{_fmt_pct(share1)}</div>
            <div style="opacity:.8;">{cand1}</div>
          </div>
          <div style="text-align:center; padding:8px; border-left:1px solid #EEF2F7;">
            <div style="display:inline-flex; padding:6px 10px; border-radius:14px; font-weight:600; color:{c2_fg}; background:{c2_bg};">{p2}</div>
            <div style="font-weight:700; margin-top:6px; color:{COLOR_TEXT_DARK};">{_fmt_pct(share2)}</div>
            <div style="opacity:.8;">{cand2}</div>
          </div>
          <div style="grid-column: 1 / -1; text-align:center; padding:10px 8px 2px; border-top:1px solid #EEF2F7;">
            <div style="color:#6B7280; font-weight:600; margin-bottom:4px;">1~2위 격차</div>
            <div style="font-weight:700; color:{COLOR_TEXT_DARK};">{gap_txt}</div>
          </div>
        </div>
        """
        from streamlit.components.v1 import html as html_component
        html_component(html, height=200, scrolling=False)

# =========================================================
# [Incumbent Card]
# NOTE(7): Reduce second-level bullet indent so long career items don't clip.
#          (margin-left decreased from 1.1rem -> 0.8rem)
# =========================================================
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
            if not s: return []
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
            <div style="font-weight:700; color:{COLOR_TEXT_DARK};">{name}</div>
            <div style="display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius:999px; font-weight:600; color:{fg}; background:{bg};">
              {party}
            </div>
          </div>
          <ul style="margin:0; padding-left:1.1rem; color:#374151;">
            <li>선수: {term}</li><li>성별: {gender}</li><li>연령: {age}</li>
            {"<li>최근 경력</li><ul style='margin:.2rem 0 0 0.8rem;'>"+items_html+"</ul>" if items_html else ""}  <!-- NOTE(7) -->
          </ul>
        </div>
        """
        from streamlit.components.v1 import html as html_component
        html_component(html, height=200, scrolling=False)

# =========================================================
# [Progressive Party Box]
# NOTE(8): Add bottom mini bar comparing local "진보 득표력" vs 10-region average.
#          Uses index_sample.csv if available; falls back gracefully.
# =========================================================
def render_prg_party_box(prg_row: pd.DataFrame|None=None, pop_row: pd.DataFrame|None=None, *, code: str|int|None=None, region: str|None=None, debug: bool=False):
    with st.container(border=True):
        st.markdown("**진보당 현황**")
        if prg_row is None or prg_row.empty:
            df_all = _load_index_df()
            if df_all is None or df_all.empty:
                st.info("지표 소스(index_sample.csv)를 찾을 수 없습니다."); return
            df_all.columns = [_norm(c) for c in df_all.columns]
            code_col = "code" if "code" in df_all.columns else None
            name_col = "region" if "region" in df_all.columns else None
            prg_row = pd.DataFrame()
            if code is not None and code_col:
                key = _norm(code); prg_row = df_all[df_all[code_col].astype(str).map(_norm)==key].head(1)
            if (prg_row is None or prg_row.empty) and region and name_col:
                key = _norm(region)
                prg_row = df_all[df_all[name_col].astype(str).map(_norm)==key].head(1)
                if prg_row.empty:
                    prg_row = df_all[df_all[name_col].astype(str).str.contains(key, na=False)].head(1)
            if prg_row is None or prg_row.empty:
                prg_row = df_all.head(1)
        else:
            df_all = _load_index_df()

        df = prg_row.copy(); df.columns = [_norm(c) for c in df.columns]; r = df.iloc[0]

        def find_col_exact_or_compact(df, prefer_name, compact_key):
            if prefer_name in df.columns: return prefer_name
            for c in df.columns:
                if compact_key in str(c).replace(" ",""): return c
            return None

        col_strength = find_col_exact_or_compact(df, "진보정당 득표력", "진보정당득표력")
        col_members  = find_col_exact_or_compact(df, "진보당 당원수", "진보당당원수")

        strength = _to_pct_float(r.get(col_strength)) if col_strength else None
        members  = _to_int(r.get(col_members)) if col_members else None

        html = f"""
        <div style="display:grid; grid-template-columns: 1fr 1fr; align-items:center; gap:12px; margin-top:6px;">
            <div style="text-align:center; padding:8px 6px;">
                <div style="color:#6B7280; font-weight:600; margin-bottom:6px;">진보 득표력</div>
                <div style="font-weight:800; color:#111827;">{_fmt_pct(strength) if strength is not None else 'N/A'}</div>
            </div>
            <div style="text-align:center; padding:8px 6px;">
                <div style="color:#6B7280; font-weight:600; margin-bottom:6px;">진보당 당원수</div>
                <div style="font-weight:800; color:#111827;">{(f"{members:,}명" if isinstance(members,int) else "N/A")}</div>
            </div>
        </div>
        """
        from streamlit.components.v1 import html as html_component
        html_component(html, height=160, scrolling=False)

        # Mini bar at bottom: local vs average (only if index_sample available)
        try:
            avg_strength = None
            if df_all is not None:
                cols_norm = [_norm(c) for c in df_all.columns]
                if col_strength and col_strength in df.columns:
                    key_cs = col_strength
                else:
                    key_cs = next((c for c in cols_norm if "진보정당득표력" in c), None)
                    if key_cs: key_cs = df_all.columns[cols_norm.index(key_cs)]
                if key_cs and key_cs in df_all.columns:
                    vals = pd.to_numeric(df_all[key_cs].astype(str).str.replace("%","", regex=False), errors="coerce")
                    avg_strength = float(vals.mean()) if vals.notna().any() else None

            if strength is not None and avg_strength is not None:
                bar_df = pd.DataFrame({
                    "항목":["해당 지역","10개 평균"],
                    "값":[strength/100.0 if strength>1 else strength, (avg_strength/100.0 if avg_strength>1 else avg_strength)]
                })
                mini = (
                    alt.Chart(bar_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("값:Q", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, max(bar_df["값"])*1.1])),
                        y=alt.Y("항목:N", title=None),
                        color=alt.condition(alt.datum.항목=="해당 지역", alt.value(COLOR_BLUE), alt.value("#9CA3AF")),
                        tooltip=[alt.Tooltip("항목:N"), alt.Tooltip("값:Q", format=".1%")]
                    )
                ).properties(height=110)
                st.altair_chart(mini, use_container_width=True, theme=None)
        except Exception:
            pass  # silent fallback

# =========================================================
# [Region Detail Layout]
# NOTE(4): Make three boxes (population_box, age_highlight_chart, sex_ratio_bar) visually same height
#          by passing the same box_height_px (=240) and keeping columns simple.
# NOTE(5): Vote trend chart called with larger height.
# =========================================================
def render_region_detail_layout(
    df_pop: pd.DataFrame | None = None,
    df_trend: pd.DataFrame | None = None,
    df_24: pd.DataFrame | None = None,
    df_cur: pd.DataFrame | None = None,
    df_prg: pd.DataFrame | None = None,
):
    st.markdown("### 👥 인구 정보")
    left, right = st.columns([1.1, 2.9])

    with left:
        render_population_box(df_pop, box_height_px=240)

    with right:
        a, b = st.columns([1.2, 2.8])
        with a.container(border=True):
            st.markdown("**연령 구성**")
            render_age_highlight_chart(df_pop, box_height_px=240, width_px=300)
        with b.container(border=True):
            st.markdown("**연령별, 성별 인구분포**")
            render_sex_ratio_bar(df_pop, box_height_px=240)

    st.markdown("### 📈 정당성향별 득표추이")
    render_vote_trend_chart(df_trend, box_height_px=420)  # NOTE(5)

    st.markdown("### 🗳️ 선거 결과 및 정치지형")
    c1, c2, c3 = st.columns(3)
    with c1:
        render_results_2024_card(df_24)
    with c2:
        render_incumbent_card(df_cur)
    with c3:
        render_prg_party_box(df_prg, df_pop)
