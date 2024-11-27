# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
import pickle

# 페이지 기본 설정
st.set_page_config(
    page_icon="🍎",
    page_title="Taurus 예측 모델"
)

# 비밀번호 설정
PASSWORD = "Taurus2024"  # 비밀번호 설정

# 비밀번호 확인 상태 초기화
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# 비밀번호 입력 화면
if not st.session_state.authenticated:
    st.markdown("<h1 style='text-align: center;'>🔒 비밀번호 입력</h1>", unsafe_allow_html=True)
    password = st.text_input('비밀번호를 입력하세요', type='password')

    if st.button("확인", use_container_width=True):
        if password == PASSWORD:
            st.session_state.authenticated = True  # 인증 성공
            st.success("비밀번호 인증 성공!")
            # 페이지 갱신 없이 메인 화면 표시
        else:
            st.error("비밀번호가 올바르지 않습니다.")

# 인증 성공 후 메인 화면
if st.session_state.authenticated:
    with open('best_ridge_model_241122.pkl', 'rb') as file:
        model = pickle.load(file)
        encoder = pickle.load(file)
        scaler = pickle.load(file)

    # 타이틀 및 설명 표시
    st.markdown(
        """
        <h1 style='text-align: center;'>Taurus 제품 용량 효율 예측 모델</h1>
        <h5 style='text-align: center;'>Taurus 공정 조건에 따른 방전 용량과 효율을 예측합니다.</h5>
        """,
        unsafe_allow_html=True
    )
    st.divider()

    # 입력값 구성
    st.markdown("""<h3 style='text-align: center;'>결과값 예측</h3>""", unsafe_allow_html=True)
    raw_material = st.selectbox('Raw Material', ['S5', 'S6', 'DS7', 'DS8', 'DS9', 'OTC'])

    input_columns = [
        'RM_PSA_D50', 'PS_Temp', 'PS_Ratio', 'p-Si_pore_volume',
        'p-Si_pore_size', 'p-Si_domain_size', 'p-Si_Oxygen',
        'C_condition', 'Carbon', 'c-Oxygen', 'c-Si_domain_size',
        'c-Surface_area', 'FCETemp'
    ]
    input_values = []


    # 세 개씩 한 행에 입력받기
    for i in range(0, len(input_columns), 3):
        cols = st.columns(3)
        for j, col in enumerate(input_columns[i:i+3]):
            with cols[j]:
                input_values.append(st.number_input(f'{col}'))


    # 예측 버튼
    if st.button('머신러닝 예측 결과', use_container_width=True):
        try:
            # raw_material 인코딩
            raw_material_encoded = encoder.transform([[raw_material]])
            raw_material_encoded_df = pd.DataFrame(
                raw_material_encoded, columns=encoder.get_feature_names_out(['raw_material'])
            )

            # 입력값 결합
            input_data_df = pd.DataFrame([input_values], columns=input_columns)
            processed_input_data = pd.concat([input_data_df, raw_material_encoded_df], axis=1)

            # 스케일링
            scaled_input_data = scaler.transform(processed_input_data)

            # 예측 수행
            predictions = model.predict(scaled_input_data)

            # 결과 출력
            st.markdown("### 예측 결과")
            results_df = pd.DataFrame({
                '예측 항목': ['방전 용량 (1V, 25℃)', '초기 효율'],
                '예측 값': [f'{predictions[0][0]:.2f} mAh/g', f'{predictions[0][1]:.2f} %']
            })
            st.table(results_df)

        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")