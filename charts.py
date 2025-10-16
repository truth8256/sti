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

def _load_index_df() -> pd.DataFrame | None:
    candidate_paths = [
        "sti/data/index_sample.csv",
        "./sti/data/index_sample.csv",
        "data/index_sample.csv",
        "./data/index_sample.csv",
        "index_sample.csv",
        "/mnt/data/index_sample.csv",
        "/mnt/data/index_sample1012.csv",
    ]
    for path in candidate_paths:
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
    
# 공통 카드 높이 (하단 세 박스 동일)
CARD_HEIGHT = 180  # 170~200 사이에서

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

# 인구 정보
def render_population_box(pop_df: pd.DataFrame):
    import numpy as np
    import altair as alt

    with st.container(border=True):
        if pop_df is None or pop_df.empty:
            st.info("유동인구/연령/성비 차트를 위한 데이터가 없습니다.")
            return

        # 컬럼 정규화
        df = pop_df.copy()
        df.columns = [str(c).strip().replace("\n", "").replace("\r", "") for c in df.columns]

        # 컬럼명 매핑
        code_col  = next((c for c in ["지역구코드","선거구코드","코드","code","CODE"] if c in df.columns), None)
        total_col = next((c for c in ["전체 유권자","전체유권자","total_voters"] if c in df.columns), None)
        float_col = next((c for c in ["유동인구","유권자 이동","floating","mobility"] if c in df.columns), None)

        if not total_col or not float_col:
            st.error("population.csv에서 '전체 유권자' 또는 '유동인구' 컬럼을 찾지 못했습니다.")
            return

        # 숫자화
        def _to_num(x):
            if pd.isna(x): return np.nan
            if isinstance(x, (int, float)): return float(x)
            s = str(x).strip().replace(",", "")
            try: return float(s)
            except: return np.nan

        df[total_col] = df[total_col].apply(_to_num)
        df[float_col] = df[float_col].apply(_to_num)

        # 구 합계 (같은 지역구코드끼리 합)
        if code_col:
            agg = df.groupby(code_col, dropna=False)[[total_col, float_col]].sum(min_count=1).reset_index(drop=True)
            total_voters = float(agg[total_col].sum())
            floating_pop = float(agg[float_col].sum())
        else:
            total_voters = float(df[total_col].sum())
            floating_pop = float(df[float_col].sum())

        if np.isnan(total_voters) and np.isnan(floating_pop):
            st.info("표시할 합계 수치가 없습니다.")
            return

        total_voters = 0.0 if np.isnan(total_voters) else total_voters
        floating_pop = 0.0 if np.isnan(floating_pop) else floating_pop

        # 유동비율
        mobility_rate = np.nan
        if total_voters > 0:
            mobility_rate = floating_pop / total_voters
        per_thousand = mobility_rate * 1000 if mobility_rate == mobility_rate else np.nan

        # 레이아웃: 좌 카드, 우 단일 비율 막대
        c1, c2 = st.columns([1, 2.2])

        with c1:
            st.markdown("**전체 유권자 수**")
            st.markdown(f"{int(round(total_voters)):,}명")

            st.markdown("**유동인구**")
            st.markdown(f"{int(round(floating_pop)):,}명")

        with c2:
            if mobility_rate == mobility_rate:  
                bar_df = pd.DataFrame({"항목": ["유동비율"], "값": [mobility_rate]})

                x_max = max(0.3, float(mobility_rate) * 1.3)  # 최소 30% 범위
                chart = (
                    alt.Chart(bar_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("값:Q",
                                axis=alt.Axis(title=None, format=".0%"),
                                scale=alt.Scale(domain=[0, x_max])),
                        y=alt.Y("항목:N", axis=alt.Axis(title=None, labels=False, ticks=False)),
                        tooltip=[alt.Tooltip("값:Q", title="유동비율", format=".1%")]
                    )
                    .properties(width=220, height=80)
                )

                # 막대 끝에 값 라벨
                text = (
                    alt.Chart(bar_df)
                    .mark_text(align="left", dx=4)
                    .encode(
                        x=alt.X("값:Q", scale=alt.Scale(domain=[0, x_max])),
                        y=alt.Y("항목:N"),
                        text=alt.Text("값:Q", format=".1%")
                    )
                )

                st.altair_chart(chart + text, use_container_width=False)
                st.caption("유동비율 = (전입 + 전출) ÷ 전체 유권자 (동일 기간 기준)")
            else:
                st.info("유동비율을 계산할 수 없습니다.")


# 연령 구성
def render_age_highlight_chart(pop_df: pd.DataFrame, *, box_height_px: int = 320):
    import numpy as np
    import altair as alt

    if pop_df is None or pop_df.empty:
        st.info("연령 구성 데이터가 없습니다.")
        return

    df = _norm_cols(pop_df.copy())

    # --- 컬럼명 ---
    Y_COL = "청년층(18~39세)"
    M_COL = "중년층(40~59세)"
    O_COL = "고령층(65세 이상)"
    TOTAL_CANDIDATES = ["전체 유권자 수", "전체 유권자", "전체유권자", "total_voters"]

    for c in (Y_COL, M_COL, O_COL):
        if c not in df.columns:
            st.error(f"필수 컬럼이 없습니다: {c}")
            return
    total_col = next((c for c in TOTAL_CANDIDATES if c in df.columns), None)
    if total_col is None:
        st.error("'전체 유권자 수' 컬럼을 찾지 못했습니다.")
        return

    # --- 숫자화 + NaN→0 (열 단위 일괄 처리) ---
    for c in (Y_COL, M_COL, O_COL, total_col):
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce",
        ).fillna(0)

    # --- 합계(안전하게 전체 합) ---
    y = float(df[Y_COL].sum())
    m = float(df[M_COL].sum())
    o = float(df[O_COL].sum())
    total_v = float(df[total_col].sum())
    if total_v <= 0:
        st.info("전체 유권자 수(분모)가 0입니다.")
        return

    # --- 60~64 계산 + 보정 ---
    a6064 = total_v - (y + m + o)
    if abs(a6064) <= 0.5:  # 소수 오차 보정
        a6064 = 0.0
    a6064 = max(0.0, min(a6064, total_v))
    denom = total_v

    # --- 데이터 테이블 ---
    labels_main = ["청년층(18~39세)", "중년층(40~59세)", "고령층(65세 이상)"]
    vals_main   = [y, m, o]
    labels_all  = labels_main + ["60~64세"]
    vals_all    = vals_main + [a6064]
    ratios_all  = [(v / denom * 100.0) if denom > 0 else 0.0 for v in vals_all]

    # --- 라디오(강조) ---
    focus = st.radio("강조", ["전체"] + labels_main, index=0, horizontal=True, label_visibility="collapsed")
    def emph_flag(name: str) -> bool:
        return (focus == "전체") or (name == focus)

    df_plot = pd.DataFrame({
        "연령": labels_all,
        "명": vals_all,
        "비율": [round(x, 6) for x in ratios_all],
        "is_extra": [False, False, False, True],  # 60~64
        "강조": [emph_flag(nm) and nm != "60~64세" for nm in labels_all],
        "순서": [0, 1, 2, 3],  # 레이어 순서 고정
    })

    # --- 색상 (명도 차 크게) ---
    color_map = {
        "청년층(18~39세)": "#82B1FF",
        "중년층(40~59세)": "#4D8EFF",
        "고령층(65세 이상)": "#1E6BFF",
        "60~64세": "#A7ADB8",
    }
    color_domain = labels_all
    color_range  = [color_map[k] for k in color_domain]

    # --- 크기/센터 ---
    width = 280
    height = max(200, box_height_px - 56)
    cx, cy = width / 2, height / 2

    base = alt.Chart(df_plot).properties(width=width, height=height)

    # 60~64 (회색, 항상 표시)
    arcs_extra = (
        base.transform_filter("datum.is_extra == true")
        .mark_arc(innerRadius=70, stroke="white", strokeWidth=1, padAngle=0.003)
        .encode(
            theta=alt.Theta("비율:Q"),
            color=alt.Color("연령:N", scale=alt.Scale(domain=color_domain, range=color_range), legend=None),
            order=alt.Order("순서:Q", sort="ascending"),
            tooltip=[alt.Tooltip("연령:N"), alt.Tooltip("명:Q", format=",.0f"), alt.Tooltip("비율:Q", format=".1f", title="비율(%)")],
        )
    )

    # 메인 3조각
    arcs_main = (
        base.transform_filter("datum.is_extra == false")
        .mark_arc(innerRadius=70, stroke="white", strokeWidth=1, padAngle=0.003)
        .encode(
            theta=alt.Theta("비율:Q"),
            color=alt.Color("연령:N", scale=alt.Scale(domain=color_domain, range=color_range),
                            legend=alt.Legend(title=None, orient="top", values=labels_main)),
            opacity=alt.condition("datum.강조 == true", alt.value(1.0), alt.value(0.25)),
            order=alt.Order("순서:Q", sort="ascending"),
            tooltip=[alt.Tooltip("연령:N"), alt.Tooltip("명:Q", format=",.0f"), alt.Tooltip("비율:Q", format=".1f", title="비율(%)")],
        )
    )

    
    # 가운데 요약
    if focus == "전체":
        big, small = f"{int(round(denom)):,}", ""
    else:
        i = labels_all.index(focus)
        big, small = f"{ratios_all[i]:.1f}%", f"{int(round(vals_all[i])):,}명"

    center_big = (
        alt.Chart(pd.DataFrame({"x":[0]}))
        .mark_text(fontSize=20, fontWeight="bold", align="center", baseline="middle", color="#0f172a")
        .encode(x=alt.value(cx), y=alt.value(cy-6), text=alt.value(big))
    )
    center_small = (
        alt.Chart(pd.DataFrame({"x":[0]}))
        .mark_text(fontSize=12, align="center", baseline="middle", color="#475569")
        .encode(x=alt.value(cx), y=alt.value(cy+14), text=alt.value(small))
    )

    st.altair_chart(arcs_extra + arcs_main + highlight + center_big + center_small, use_container_width=False)

# 정당성향별 득표추이
def render_vote_trend_chart(ts: pd.DataFrame):
    if ts is None or ts.empty:
        st.info("득표 추이 데이터가 없습니다.")
        return

    import re
    df = _norm_cols(ts.copy())

    # ---- 컬럼 감지
    label_col = next((c for c in ["계열","성향","정당성향","party_label","label"] if c in df.columns), None)
    value_col = next((c for c in ["득표율","비율","share","ratio","pct","prop"] if c in df.columns), None)
    wide_value_cols = [c for c in ["민주","보수","진보","기타"] if c in df.columns]

    # 선거 식별 후보 (우선순위 명시)
    prefer_ids = ["선거명","election","선거","분류","연도","year"]
    fallback_ids = ["코드","code"]
    id_col = next((c for c in prefer_ids if c in df.columns), None)
    if id_col is None:
        id_col = next((c for c in fallback_ids if c in df.columns), None)

    year_col = next((c for c in ["연도","year"] if c in df.columns), None)

    # ---- wide → long
    if wide_value_cols:
        if not id_col:
            st.warning("선거명을 식별할 컬럼이 필요합니다. (선거명/election/연도/코드)")
            return
        long_df = df.melt(id_vars=id_col, value_vars=wide_value_cols,
                          var_name="계열", value_name="득표율")
        base_elec = long_df[id_col].astype(str)
    else:
        if not (label_col and value_col):
            st.warning("정당 성향(계열)과 득표율 컬럼이 필요합니다.")
            return
        long_df = df.rename(columns={label_col:"계열", value_col:"득표율"}).copy()
        if id_col:
            base_elec = long_df[id_col].astype(str)
        elif year_col:
            base_elec = long_df[year_col].astype(str)
        else:
            st.warning("선거명을 식별할 컬럼이 필요합니다. (선거명/election/연도/코드)")
            return

    # ---- 선거명 한글화 (지역/체계 보존)
    def _norm_token(s: str) -> str:
        s = str(s).strip().replace("-", "_").replace(" ", "_").upper()
        s = re.sub(r"_+", "_", s)
        return s

    CODE_RE = re.compile(r"^(20\d{2})(?:_([SG]))?_(NA|LOC|PRESIDENT)(?:_(PRO|GOV))?$")
    KR_REGION_RE = re.compile(r"^(20\d{2})\s+(서울|경기)\s+(.*)$")

    def to_kr_label(raw: str) -> str:
        s = str(raw)
        key = _norm_token(s)
        m = CODE_RE.fullmatch(key)
        if m:
            year, region_tag, lvl, kind = m.group(1), m.group(2), m.group(3), m.group(4)
            region_txt = f" {region_tag} " if region_tag else " "
            if lvl == "PRESIDENT":
                return f"{year}{region_txt}대선".strip()
            if lvl == "NA" and (kind == "PRO"):
                return f"{year}{region_txt}총선 비례".strip()
            if lvl == "LOC" and (kind == "PRO"):
                return f"{year}{region_txt}광역 비례".strip()
            if lvl == "LOC" and (kind == "GOV"):
                return f"{year}{region_txt}광역단체장".strip()
        # 한글형: 지역 접두부 보존
        km = KR_REGION_RE.match(s)
        if km:
            return f"{km.group(1)} {km.group(2)} {km.group(3)}"
        if re.match(r"^\s*20\d{2}", s):
            return s.strip()
        return s

    long_df["선거명_표시"] = base_elec.apply(to_kr_label)

    # ---- (득표율 변환 제거) 데이터 그대로 사용
    # long_df["득표율"] = long_df["득표율"].apply(_to_pct)

    long_df = long_df.dropna(subset=["선거명_표시","계열","득표율"])

    # ---- 연도 추출 & 정렬 고정
    long_df["연도정렬"] = long_df["선거명_표시"].str.extract(r"^(20\d{2})").astype(int)
    long_df = long_df.sort_values(["연도정렬","선거명_표시","계열"])

    # ---- 중복 제거 (평균 처리 제거)
    long_df = long_df.drop_duplicates(subset=["선거명_표시","연도정렬","계열","득표율"])

    election_order = long_df.sort_values(["연도정렬","선거명_표시"])["선거명_표시"].unique().tolist()

    # 색상
    party_order = ["민주","보수","진보","기타"]
    color_map = {"민주":"#152484", "보수":"#E61E2B", "진보":"#7B2CBF", "기타":"#6C757D"}
    present = [p for p in party_order if p in long_df["계열"].unique().tolist()]
    colors  = [color_map[p] for p in present]

    # ---- Altair: 넓은 히트박스 + 인터랙티브 줌/팬
    import altair as alt
    selector = alt.selection_point(fields=["선거명_표시","계열"], nearest=True, on="mouseover", empty=False)

    line = (
        alt.Chart(long_df)
        .mark_line(point=False, strokeWidth=3)
        .encode(
            x=alt.X("선거명_표시:N",
                    sort=election_order,
                    title="선거명",
                    axis=alt.Axis(labelAngle=-35, labelOverlap=False, labelPadding=6, labelLimit=280)),
            y=alt.Y("득표율:Q", title="득표율(%)"),
            color=alt.Color("계열:N", scale=alt.Scale(domain=present, range=colors),
                            legend=alt.Legend(title=None, orient="top")),
        )
    )

    hit = (
        alt.Chart(long_df)
        .mark_circle(size=600, opacity=0)
        .encode(x="선거명_표시:N", y="득표율:Q", color=alt.Color("계열:N", legend=None))
        .add_params(selector)
    )

    points = (
        alt.Chart(long_df)
        .mark_circle(size=140)
        .encode(
            x="선거명_표시:N",
            y="득표율:Q",
            color=alt.Color("계열:N", scale=alt.Scale(domain=present, range=colors), legend=None),
            opacity=alt.condition(selector, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip("선거명_표시:N", title="선거명"),
                alt.Tooltip("계열:N", title="계열"),
                alt.Tooltip("득표율:Q", title="득표율(%)", format=".1f"),
            ],
        )
        .transform_filter(selector)
    )

    chart = (line + hit + points).properties(height=360).interactive()

    with st.container(border=True):
        st.altair_chart(chart, use_container_width=True)

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
        html_component(html, height=CARD_HEIGHT, scrolling=False)

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
        from streamlit.components.v1 import html as html_component
        html_component(html, height=CARD_HEIGHT, scrolling=False)

# 진보당 현황
def render_prg_party_box(
    prg_row: pd.DataFrame | None,
    pop_row: pd.DataFrame | None = None,
    *,
    code: str | int | None = None,
    region: str | None = None,
    debug: bool = False
):
    def _norm(s: str) -> str:
        s = str(s).replace("\n", " ").replace("\r", " ").strip()
        return " ".join(s.split())

    with st.container(border=True):
        st.markdown("**진보당 현황**")
        st.markdown("<div style='padding-top:4px;'></div>", unsafe_allow_html=True)

        # ---- 데이터 확보 ----
        if prg_row is None or prg_row.empty:
            df_all = _load_index_df()
            if df_all is None or df_all.empty:
                st.info("지표 소스(index_sample.csv)를 찾을 수 없습니다. (sti/data/index_sample.csv 경로 확인)")
                return
            df_all.columns = [_norm(c) for c in df_all.columns]
            code_col = "code" if "code" in df_all.columns else None
            name_col = "region" if "region" in df_all.columns else None

            prg_row = pd.DataFrame()
            if code is not None and code_col:
                key = _norm(code); prg_row = df_all[df_all[code_col].astype(str).map(_norm) == key].head(1)
            if (prg_row is None or prg_row.empty) and region and name_col:
                key = _norm(region)
                prg_row = df_all[df_all[name_col].astype(str).map(_norm) == key].head(1)
                if prg_row.empty:
                    prg_row = df_all[df_all[name_col].astype(str).str.contains(key, na=False)].head(1)
            if prg_row is None or prg_row.empty:
                prg_row = df_all.head(1)

        df = prg_row.copy()
        df.columns = [_norm(c) for c in df.columns]
        r = df.iloc[0]

        # 정확 컬럼 매칭
        col_strength = "진보정당 득표력" if "진보정당 득표력" in df.columns else \
                       next((c for c in df.columns if "진보정당득표력" in c.replace(" ", "")), None)
        col_members  = "진보당 당원수"   if "진보당 당원수"   in df.columns else \
                       next((c for c in df.columns if "진보당당원수"   in c.replace(" ", "")), None)

        strength = _to_pct_float(r.get(col_strength)) if col_strength else None  # 0~100 가정
        members  = _to_int(r.get(col_members)) if col_members else None

        # ---- 대시보드형 ----
        from streamlit.components.v1 import html as html_component
        
        # 좌/우 2칸 그리드 + 적당한 여백
        html = f"""
        <div style="
          display:grid; grid-template-columns: 1fr 1fr;
          align-items:center; gap:12px; margin-top:6px;">
          <div style="text-align:center; padding:8px 6px;">
            <div style="color:#6B7280; font-weight:600; font-size:0.95rem; margin-bottom:6px;">진보 득표력</div>
            <div style="font-weight:800; font-size:1.20rem; color:#111827;
                        letter-spacing:-0.2px; font-variant-numeric:tabular-nums;">
              {_fmt_pct(strength)}
            </div>
          </div>
          <div style="text-align:center; padding:8px 6px;">
            <div style="color:#6B7280; font-weight:600; font-size:0.95rem; margin-bottom:6px;">진보당 당원수</div>
            <div style="font-weight:800; font-size:1.20rem; color:#111827;
                        letter-spacing:-0.2px; font-variant-numeric:tabular-nums;">
              { (f"{members:,}명" if isinstance(members,(int,float)) and members is not None else "N/A") }
            </div>
          </div>
        </div>
        """
        html_component(html, height=CARD_HEIGHT, scrolling=False)

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
    
    with top_left:
        render_population_box(df_pop)
    
    with top_right:
        subcol1, subcol2 = st.columns(2)
        with subcol1.container(border=True):
            st.markdown("#### 연령 구성")
            render_age_highlight_chart(df_pop, box_height_px=320)
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










