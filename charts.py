# =============================
# File: charts.py
# =============================
from __future__ import annotations

import re
import pandas as pd
import streamlit as st
import altair as alt
from metrics import compute_24_gap

# -------- 유틸 --------
def _to_pct_float(v, default=None):
    # '45.2%', '45,2', 0.452, ' 45.2 % ' -> 45.2
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

# -------- 내부: 파이차트 생성 (Altair) --------
def _pie_chart(title: str, labels: list[str], values: list[float], colors: list[str], width: int = 260, height: int = 260):
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

# -------- 24년 결과 카드 (5_na_dis_results.csv) --------
def render_results_2024_card(res_row: pd.DataFrame, df_24: pd.DataFrame = None, code: str = None):
    # 여러 연도 행이 들어오므로, 2024가 있으면 2024, 없으면 최댓값 연도를 선택
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

    # 이 데이터셋은 '후보1_이름/후보1_득표율/후보2_...' 형식(5_na_dis_results.csv)
    c1n = next((c for c in ["후보1_이름", "1위이름", "1위 후보", "1위_이름", "1st_name"] if c in res_row.columns), None)
    c1v = next((c for c in ["후보1_득표율", "1위득표율", "1위 득표율", "1st_share", "1위득표율(%)"] if c in res_row.columns), None)
    c2n = next((c for c in ["후보2_이름", "2위이름", "2위 후보", "2위_이름", "2nd_name"] if c in res_row.columns), None)
    c2v = next((c for c in ["후보2_득표율", "2위득표율", "2위 득표율", "2nd_share", "2위득표율(%)"] if c in res_row.columns), None)

    name1 = str(r.get(c1n)) if c1n else "1위"
    share1 = _to_pct_float(r.get(c1v))
    name2 = str(r.get(c2n)) if c2n else "2위"
    share2 = _to_pct_float(r.get(c2v))

    gap = None
    if isinstance(share1, (int, float)) and isinstance(share2, (int, float)):
        gap = round(share1 - share2, 2)
    elif df_24 is not None and code is not None:
        gap = compute_24_gap(df_24, code)

    box = st.container()
    with box:
        st.markdown("**24년 총선결과**")
        col1, col2, col3 = st.columns([1.2, 1.2, 1])
        with col1:
            st.metric(label=f"{name1}", value=_fmt_pct(share1))
        with col2:
            st.metric(label=f"{name2}", value=_fmt_pct(share2))
        with col3:
            st.metric(label="1~2위 격차", value=_fmt_gap(gap))

# -------- 현직 정보 카드 (current_info.csv) --------
def render_incumbent_card(cur_row: pd.DataFrame):
    if cur_row is None or cur_row.empty:
        st.info("현직 정보 데이터가 없습니다.")
        return

    cur_row = _norm_cols(cur_row)
    r = cur_row.iloc[0]
    name_col   = next((c for c in ["의원명", "이름", "성명", "incumbent_name"] if c in cur_row.columns), None)
    party_col  = next((c for c in ["정당", "소속정당", "party"] if c in cur_row.columns), None)
    term_col   = next((c for c in ["선수", "당선횟수", "terms"] if c in cur_row.columns), None)
    age_col    = next((c for c in ["연령", "나이", "age"] if c in cur_row.columns), None)
    gender_col = next((c for c in ["성별", "gender"] if c in cur_row.columns), None)
    status_col = next((c for c in ["상태", "현직여부", "status"] if c in cur_row.columns), None)

    box = st.container()
    with box:
        st.markdown("**현직정보**")
        st.write(f"- 의원: **{r.get(name_col, 'N/A')}** / 정당: **{r.get(party_col, 'N/A')}**")
        st.write(
            f"- 선수: **{r.get(term_col, 'N/A')}** / 성별: **{r.get(gender_col, 'N/A')}** / 연령: **{r.get(age_col, 'N/A')}**"
        )
        if status_col:
            st.caption(f"상태: {r.get(status_col)}")

# -------- 진보당 현황 박스 (party_labels.csv 기반 - 컬럼 유연 처리) --------
def render_prg_party_box(prg_row: pd.DataFrame, pop_row: pd.DataFrame):
    """
    party_labels.csv는 원래 '정당 코드/라벨/계열' 매핑 성격.
    - 만약 '진보당 득표력' 같은 지표가 있으면 표시
    - 없으면 '지표 미제공'로 안전 처리
    - 조직/후보 수 같은 필드는 있으면 표시
    """
    box = st.container()
    with box:
        st.markdown("**진보당 현황**")
        if prg_row is None or prg_row.empty:
            st.info("진보당 관련 데이터가 없습니다.")
            return

        prg_row = _norm_cols(prg_row)
        r = prg_row.iloc[0]

        # 득표력/조직/후보 컬럼 유연 탐색
        strength_col = next((c for c in ["진보당 득표력","득표력","progressive_strength","PL_prg_str"] if c in prg_row.columns), None)
        org_col      = next((c for c in ["진보당 당원수","당원수","조직수","branch_count","members"] if c in prg_row.columns), None)
        cand_col     = next((c for c in ["진보당 지방선거후보","지방선거후보수","local_candidates"] if c in prg_row.columns), None)

        c1, c2 = st.columns(2)
        with c1:
            if strength_col and pd.notna(r.get(strength_col)):
                st.metric("진보득표력", _fmt_pct(_to_pct_float(r.get(strength_col))))
            else:
                st.metric("진보득표력", "지표 미제공")
        with c2:
            st.metric("조직 규모", f"{_to_int(r.get(org_col)):,}" if org_col and pd.notna(r.get(org_col)) else "N/A")

        if cand_col and pd.notna(r.get(cand_col)):
            st.caption(f"지방선거 후보 수: {_to_int(r.get(cand_col)):,}명")

        # 인구 맥락 간단 표시
        if pop_row is not None and not pop_row.empty:
            pop_row = _norm_cols(pop_row)
            rp = pop_row.iloc[0]
            elder_col = next((c for c in ["고령층비율", "65세이상비율", "age65p"] if c in pop_row.columns), None)
            youth_col = next((c for c in ["청년층비율", "39세이하비율", "age39m"] if c in pop_row.columns), None)
            with st.expander("인구 맥락 보기", expanded=False):
                elder = _fmt_pct(_to_pct_float(rp.get(elder_col))) if elder_col and pd.notna(rp.get(elder_col)) else "N/A"
                youth = _fmt_pct(_to_pct_float(rp.get(youth_col))) if youth_col and pd.notna(rp.get(youth_col)) else "N/A"
                st.write(f"- 고령층 비율: {elder} / 청년층 비율: {youth}")

# -------- 득표 추이 차트 (vote_trend.csv) --------
def render_vote_trend_chart(ts: pd.DataFrame):
    if ts is None or ts.empty:
        st.info("득표 추이 데이터가 없습니다.")
        return

    df = _norm_cols(ts)

    # CASE A) long 포맷: election/label/prop (+ year)
    if {"label", "prop"}.issubset(df.columns) and (("election" in df.columns) or ("year" in df.columns) or ("연도" in df.columns)):
        if "year" not in df.columns:
            if "election" in df.columns:
                df["year"] = df["election"].astype(str).str.extract(r"(\d{4})")[0].astype("Int64")
            elif "연도" in df.columns:
                df["year"] = pd.to_numeric(df["연도"], errors="coerce")
        df["prop"] = pd.to_numeric(df["prop"], errors="coerce")
        df = df.dropna(subset=["year", "prop"])
        if df.empty:
            st.info("그릴 수 있는 득표 데이터가 없습니다.")
            return

    # CASE B) wide 포맷: year + 각 성향 컬럼들 → melt
    elif ("year" in df.columns or "연도" in df.columns):
        if "year" not in df.columns:
            df["year"] = pd.to_numeric(df["연도"], errors="coerce")
        value_cols = [c for c in df.columns if c not in ["year","연도"]]
        if not value_cols:
            st.info("득표 성향 컬럼이 없어 차트를 그릴 수 없습니다.")
            return
        df = df[["year"] + value_cols].copy()
        df = df.melt(id_vars=["year"], var_name="label", value_name="prop")
        df["prop"] = pd.to_numeric(df["prop"], errors="coerce")
        df = df.dropna(subset=["year","prop"])
        if df.empty:
            st.info("그릴 수 있는 득표 데이터가 없습니다.")
            return
    else:
        st.warning("vote_trend 데이터에 필요한 컬럼(연도/성향/득표)이 부족합니다.")
        st.dataframe(df.head())
        return

    # 범례 순서 & 색상 (필요 시 bookmark.csv/party_labels.csv 기준으로 교체 가능)
    party_order  = ["민주", "보수", "진보", "기타"]
    party_colors = ["#152484", "#E61E2B", "#450693", "#798897"]

    vmax = df["prop"].max()
    y_enc = alt.Y("prop:Q", title="득표율(%)") if vmax > 1 else alt.Y("prop:Q", title="득표율", axis=alt.Axis(format=".0%"))

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="연도", sort="ascending"),
            y=y_enc,
            color=alt.Color(
                "label:N",
                title="정당계열",
                scale=alt.Scale(domain=party_order, range=party_colors),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip("year:O", title="연도"),
                alt.Tooltip("label:N", title="계열"),
                alt.Tooltip("prop:Q", title="득표", format=".2f"),
            ],
        )
        .properties(height=300)
    )

    st.altair_chart(chart, use_container_width=True)

# -------- 인구 정보 박스 (population.csv) --------
def render_population_box(pop_df: pd.DataFrame):
    box = st.container()
    with box:
        st.markdown("**인구 정보**")
 
        if pop_df is None or pop_df.empty:
            st.info("인구 데이터가 없습니다.")
            return
 
        pop_df = _norm_cols(pop_df)
        r = pop_df.iloc[0]
 
        # 1) 비율 컬럼 우선
        elder_col  = next((c for c in ["고령층비율", "65세이상비율", "age65p"] if c in pop_df.columns), None)
        youth_col  = next((c for c in ["청년층비율", "39세이하비율", "age39m"] if c in pop_df.columns), None)
        mid_col    = next((c for c in ["40_59비율", "40-59비율", "age40_59p", "4050비율"] if c in pop_df.columns), None)
 
        male_col   = next((c for c in ["남성비율", "남", "male_p", "2030 남성비율"] if c in pop_df.columns), None)
        female_col = next((c for c in ["여성비율", "여", "female_p", "2030 여성비율"] if c in pop_df.columns), None)
 
        elder_pct = _to_pct_float(r.get(elder_col)) if elder_col else None
        youth_pct = _to_pct_float(r.get(youth_col)) if youth_col else None
        mid_pct   = _to_pct_float(r.get(mid_col))   if mid_col   else None
        male_pct  = _to_pct_float(r.get(male_col))  if male_col  else None
        female_pct= _to_pct_float(r.get(female_col))if female_col else None
 
        # 2) 인원수 기반 비율
        total_col = next((c for c in ["유권자수", "유권자 수", "voters", "전체 유권자"] if c in pop_df.columns), None)
        c2030_col = "2030" if "2030" in pop_df.columns else None
        c4050_col = "4050" if "4050" in pop_df.columns else None
        c65p_col  = "65세 이상" if "65세 이상" in pop_df.columns else None
 
        total = _to_int(r.get(total_col)) if total_col else None
        v2030 = _to_int(r.get(c2030_col)) if c2030_col else None
        v4050 = _to_int(r.get(c4050_col)) if c4050_col else None
        v65p  = _to_int(r.get(c65p_col))  if c65p_col  else None
 
        def pct(val):
            return (val / total * 100.0) if (isinstance(val, (int, float)) and isinstance(total, (int, float)) and total) else None
 
        if elder_pct is None:
            elder_pct = pct(v65p)
        if youth_pct is None:
            youth_pct = pct(v2030)
        if mid_pct is None:
            mid_pct = pct(v4050)
 
        # 60-64 추가(남는 비율로 추정)
        s_pct = None
        if all(isinstance(x, (int, float)) for x in [youth_pct, mid_pct, elder_pct]) and isinstance(total, (int, float)):
            used = youth_pct + mid_pct + elder_pct
            s_pct = max(0.0, 100.0 - used)
 
        # 2030 남/여
        male_share_2030 = None
        female_share_2030 = None
        has_2030_m = "2030 남성" in pop_df.columns and pd.notna(r.get("2030 남성"))
        has_2030_f = "2030 여성" in pop_df.columns and pd.notna(r.get("2030 여성"))
        if (has_2030_m or has_2030_f) and v2030:
            m_cnt = _to_int(r.get("2030 남성")) if has_2030_m else 0
            f_cnt = _to_int(r.get("2030 여성")) if has_2030_f else 0
            denom = (m_cnt + f_cnt) if (m_cnt + f_cnt) > 0 else v2030
            male_share_2030 = (m_cnt / denom * 100.0) if denom else None
            female_share_2030 = (f_cnt / denom * 100.0) if denom else None
        else:
            male_share_2030 = male_pct
            female_share_2030 = female_pct
 
        # --- 상단: 유권자 수 ---
        st.metric("유권자 수", f"{total:,}" if isinstance(total, (int, float)) else "N/A")
 
        # --- 파이차트 2개 ---
        col1, col2 = st.columns(2)
 
        y = youth_pct or 0.0
        m = mid_pct   or 0.0
        s = s_pct     or 0.0
        e = elder_pct or 0.0
        with col1:
            age_colors = ["#deebf7", "#9ecae1", "#6baed6", "#08519c"]
            _pie_chart("연령 구성", ["청년층(≤39)", "40-59", "60-64", "65+"], [y, m, s, e], colors=age_colors)
 
        mm = male_share_2030 or 0.0
        ff = female_share_2030 or 0.0
        with col2:
            if mm == 0 and ff == 0:
                st.info("2030 남/여 자료가 없습니다.")
            else:
                gender_colors = ["#bdd7e7", "#08519c"]
                _pie_chart("2030 성별 구성", ["남성", "여성"], [mm, ff], colors=gender_colors)
