# =============================
# File: app.py
# =============================
from __future__ import annotations
import streamlit as st
import pandas as pd
from pathlib import Path

# 내부 모듈
from data_loader import (
    load_population_agg,     # population.csv
    load_party_labels,       # party_labels.csv
    load_vote_trend,         # vote_trend.csv
    load_results_2024,       # 5_na_dis_results.csv
    load_current_info,       # current_info.csv
    load_index_sample,       # index_sample1012.csv
)
from metrics import (
    compute_trend_series,
    compute_summary_metrics,
    compute_24_gap,
)
from charts import (
    render_region_detail_layout,
)

# =============================
# 페이지 설정
# =============================
st.set_page_config(
    page_title="지역구 선정 1단계 조사 결과 대시보드",
    page_icon="🗳️",
    layout="wide"
)

# =============================
# 타이틀 및 헤더
# =============================
st.markdown(
    """
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <h1 style="margin:0;">🗳️ 지역구 선정 1단계 조사 결과 대시보드</h1>
        <span style="font-size:0.9rem; color:gray;">에스티아이</span>
    </div>
    """,
    unsafe_allow_html=True
)

# =============================
# Sidebar
# =============================
st.sidebar.header("메뉴 선택")
menu = st.sidebar.radio("메뉴", ["개요", "지역별 분석"])

# =============================
# 데이터 로드
# =============================
data_dir = Path("/mnt/data")

try:
    df_pop = load_population_agg(data_dir)
    df_trend = load_vote_trend(data_dir)
    df_24 = load_results_2024(data_dir)
    df_cur = load_current_info(data_dir)
    df_prg = load_party_labels(data_dir)
    df_idx = load_index_sample(data_dir)
except Exception as e:
    st.error(f"데이터 로드 중 오류 발생: {e}")
    st.stop()

# =============================
# 페이지 라우팅
# =============================
if menu == "개요":
    st.markdown("### 🧭 개요 페이지 준비중입니다.")
    st.info("이곳에는 전체 10개 선거구의 요약 통계 및 비교지표가 들어갈 예정입니다.")

elif menu == "지역별 분석":
    # 첫 화면: 지역 선택 안내
    st.markdown("### 📍 지역별 분석")
    if df_pop is None or df_pop.empty:
        st.warning("인구 데이터가 없습니다. 파일을 확인하세요.")
    else:
        region_list = sorted(df_pop["지역"].dropna().unique().tolist()) if "지역" in df_pop.columns else []
        selected_region = st.sidebar.selectbox("지역 선택", [""] + region_list, index=0)

        if not selected_region:
            st.write("#### 지역을 선택하세요.")
        else:
            # 선택된 지역 타이틀
            st.subheader(f"📊 {selected_region}")
            # 제목 우측 상단에 원래 타이틀 배치
            st.markdown(
                """
                <style>
                div[data-testid="stHeader"] {display:none;}
                </style>
                """,
                unsafe_allow_html=True
            )

            # 해당 지역 데이터 필터링 (안전하게)
            pop_sel = df_pop[df_pop["지역"] == selected_region] if "지역" in df_pop.columns else None
            trend_sel = df_trend[df_trend["지역"] == selected_region] if "지역" in df_trend.columns else df_trend
            res_sel = df_24[df_24["지역"] == selected_region] if "지역" in df_24.columns else df_24
            cur_sel = df_cur[df_cur["지역"] == selected_region] if "지역" in df_cur.columns else df_cur
            prg_sel = df_prg[df_prg["지역"] == selected_region] if "지역" in df_prg.columns else df_prg

            # 상세 레이아웃 렌더링
            render_region_detail_layout(
                df_pop=pop_sel,
                df_trend=trend_sel,
                df_24=res_sel,
                df_cur=cur_sel,
                df_prg=prg_sel
            )
