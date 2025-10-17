import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Smart Water Control", layout="wide")

st.title("💧 Smart Water Control System")
st.write("نظام ذكي للتنبؤ والتحكم في جرعات الكلور وفتح وغلق الصمامات باستخدام الذكاء الاصطناعي")

# تحميل النماذج
chlorine_model = joblib.load("chlorine_rf_model.pkl")
valve_model = joblib.load("valve_rf_model.pkl")

# تقسيم الواجهة
tab1, tab2 = st.tabs(["🔹 التنبؤ بالكلور", "🔸 التحكم بالصمامات"])

# التبويب الأول
with tab1:
    st.header("🔹 التنبؤ بتركيز الكلور")
    pressure = st.number_input("Pressure (bar)", 1.0, 4.0, 2.0)
    demand = st.number_input("Demand (m³/h)", 0.0, 500.0, 100.0)
    age = st.number_input("Water Age (hr)", 0.0, 24.0, 5.0)
    pattern = st.number_input("Pattern Factor", 0.9, 1.1, 1.0)

    if st.button("تنفيذ التنبؤ بالكلور"):
        X = pd.DataFrame([[pressure, demand, age, pattern]],
                         columns=["Pressure (bar)", "Demand (m³/h)", "Age (hr)", "Pattern_Factor"])
        y_pred = chlorine_model.predict(X)[0]
        st.success(f"تركيز الكلور المتوقع = {y_pred:.3f} mg/L")

# التبويب الثاني
with tab2:
    st.header("🔸 التنبؤ بنسبة فتح وغلق الصمامات")
    pressure = st.number_input("Pressure (bar)", 1.0, 6.0, 3.0, key="p2")
    hour = st.slider("Hour", 0, 23, 12)
    scenario = st.selectbox("Scenario", ["Peak", "Minimum"])

    if st.button("تنفيذ التنبؤ بالصمامات"):
        X = pd.DataFrame([[pressure, scenario, hour]],
                         columns=["Pressure (bar)", "Scenario", "Hour"])
        X = pd.get_dummies(X, columns=["Scenario"], drop_first=True)
        y_close = valve_model.predict(X)[0]
        y_open = 100 - y_close
        st.info(f"نسبة الغلق = {y_close:.2f}% | نسبة الفتح = {y_open:.2f}%")
