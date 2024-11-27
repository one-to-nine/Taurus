# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings
import statsmodels.api as sm
warnings.filterwarnings('ignore')

# 페이지 기본 설정
st.set_page_config(
    page_icon="🍎",
    page_title="Taurus 예측 모델"
)

# 비밀번호 인증 여부 확인
if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.warning("🔒 이 페이지에 접근하려면 먼저 비밀번호를 입력하세요.")
    st.stop()  # 페이지 로딩 중단


# 타이틀 및 설명 표시
st.markdown(
        """
        <h1 style='text-align: center;'>데이터 시각화</h1>
        """,
        unsafe_allow_html=True)

data = pd.read_csv('Taurus_240820_2.csv')

if st.checkbox('데이터 확인'):
    st.write(data)

if 'Sample ID' in data.columns:
    data = data.drop(columns=['Sample ID'])   

result_columns = ['Carbon', 'LiCap', 'DeliCap', 'FCE']
categorical_columns = ['raw_material', 'PS_Temp', 'C_condition']

# 탭 생성
tab1, tab2, tab3, tab4, tab5 = st.tabs(["파이 플롯 (Pie Plot)", "박스 플롯 (Box Plot)", "산점도 (Scatter Plot)", "히트맵 (Heatmap)", "3차원 산점도 (3D Scatter Plot)"])

with tab1:
    st.markdown(
        "<h3 style='text-align: center;'>원료별 구성 비율</h3>", 
        unsafe_allow_html=True
    )
    count_data = data['raw_material'].value_counts().reset_index().sort_values(by='count', ascending=False)
    count_data.columns = ['원료명', 'count']
    fig1 = px.pie(count_data, names='원료명', values='count', hole=0.5)
    fig1.update_traces(text = count_data['원료명'], textposition = 'outside', textfont_size=15)
    st.plotly_chart(fig1)

with tab2:
    st.markdown(
        "<h3 style='text-align: center;'>데이터 분포</h3>", 
        unsafe_allow_html=True
    )
    col1, col2 = st.columns(2)
    with col1:
        x_column = st.selectbox("x축", categorical_columns)
    with col2:
        y_column = st.selectbox("y축", result_columns)
    fig2 = px.box(data, x=x_column, y=y_column, color=x_column, color_discrete_sequence=px.colors.qualitative.Set2)
    y_min = data[y_column].min()
    y_max = data[y_column].max()
    y_range = [y_min * 0.95, y_max * 1.05]
    fig2.update_yaxes(range=y_range)
    st.plotly_chart(fig2)

with tab3:
    st.markdown(
        "<h3 style='text-align: center;'>두 변수 간의 상관관계</h3>", 
        unsafe_allow_html=True
    )
    col3, col4 = st.columns(2)
    with col3:
        x_scatter_column = st.selectbox("x축", [col for col in data.columns if col != 'raw_material'])
    with col4:
        y_scatter_column = st.selectbox("y축", [col for col in data.columns if col != 'raw_material'])

    fig3 = px.scatter(data, x=x_scatter_column, y=y_scatter_column, trendline='ols')
    fig3.update_traces(marker_size = 12, marker_color = 'lightcoral')

    # R2 값 계산
    x = data[x_scatter_column]
    y = data[y_scatter_column]
    x = sm.add_constant(x)  # 상수항 추가
    model = sm.OLS(y, x).fit()

    r2 = model.rsquared
    st.plotly_chart(fig3)
    st.write(f":pushpin: R2 값: {r2:.2f}")

with tab4:
    st.markdown(
        "<h3 style='text-align: center;'>두 변수 간의 상관관계 시각화</h3>", 
        unsafe_allow_html=True
    )
    corr_matrix = data.drop(columns=['raw_material']).corr()
    fig4 = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmin=-1, zmax=1
    ))
    fig4.update_layout(title='Correlation Heatmap', xaxis_nticks=36)
    st.plotly_chart(fig4)

    # 상관계수가 1이 아닌 0.6 이상인 항목들 필터링 및 표로 출력
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= 0.6 and corr_matrix.iloc[i, j] != 1:
                high_corr_pairs.append((corr_matrix.index[i], corr_matrix.columns[j], round(corr_matrix.iloc[i, j], 2)))

    # 데이터프레임으로 변환하여 절대값 기준 오름차순으로 정렬 후 인덱스 재배열 및 표 출력
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs, columns=['변수 1', '변수 2', '상관계수'])
        high_corr_df['상관계수_절대값'] = high_corr_df['상관계수'].abs()
        high_corr_df = high_corr_df.sort_values(by='상관계수_절대값', ascending=False).drop(columns=['상관계수_절대값']).reset_index(drop=True)
        if st.checkbox("상관계수가 0.6 이상인 변수 쌍"):
            st.write(high_corr_df)
    else:
        st.write("상관계수가 0.6 이상인 변수 쌍이 없습니다.")

with tab5:
    st.markdown(
        "<h3 style='text-align: center;'>세 변수 간의 상관관계</h3>", 
        unsafe_allow_html=True
    )
    col5, col6, col7 = st.columns(3)
    with col5:
        x_3d = st.selectbox("x축", [col for col in data.columns if col != 'raw_material'], key='3dscatter_x')
    with col6:
        y_3d = st.selectbox("y축", [col for col in data.columns if col != 'raw_material'], key='3dscatter_y')
    with col7:
        z_3d = st.selectbox("z축", [col for col in data.columns if col != 'raw_material'], key='3dscatter_z')

    fig5 = px.scatter_3d(data, x=x_3d, y=y_3d, z=z_3d)
    fig5.update_traces(marker_size = 5)
    st.plotly_chart(fig5)