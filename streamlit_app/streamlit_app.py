"""Thin Streamlit front-end. All intelligence lives in the API; this file only collects inputs and shows the answer.

    streamlit run streamlit_app/app.py
Set KLIA_API_URL (default http://localhost:8000) and, if you enabled it, KLIA_API_KEY.
"""
import datetime as dt
import os

import requests
import streamlit as st

# st.secrets takes precedence (Streamlit Cloud); fall back to env vars (local).
API = st.secrets.get("KLIA_API_URL", os.environ.get("KLIA_API_URL", "http://localhost:8000")).rstrip("/")
_key = st.secrets.get("KLIA_API_KEY", os.environ.get("KLIA_API_KEY", ""))
HEADERS = {"X-API-Key": _key} if _key else {}

st.set_page_config(page_title="AeroPredict KLIA", page_icon="✈️")
st.title("✈️ AeroPredict KLIA")


@st.cache_data(ttl=600, show_spinner=False)
def options():
    r = requests.get(f"{API}/v1/options", headers=HEADERS, timeout=60)   # first call may wake a sleeping host
    r.raise_for_status()
    return r.json()


try:
    with st.spinner("Connecting to prediction service..."):
        opt = options()
except Exception as e:
    st.error(f"Cannot reach the API at {API}: {e}")
    st.stop()

# Deduplicate airline names: strip everything in brackets
airlines_raw = opt["airlines"]
airlines_dedup = sorted(set(a.split("(")[0].strip() for a in airlines_raw))

airline = st.selectbox("Airline", airlines_dedup)
dests = opt["destinations"]
destination = st.selectbox("Destination", dests)
c1, c2 = st.columns(2)
day = c1.date_input("Date", dt.date.today())
time_ = c2.time_input("Scheduled departure (KLIA local time)", dt.time(9, 0))
aircraft = st.text_input("Aircraft (optional, e.g. A320)")

if st.button("Predict", type="primary"):
    body = {"airline": airline, "destination": destination,
            "scheduled_departure": dt.datetime.combine(day, time_).isoformat(), "aircraft": aircraft or None}
    try:
        r = requests.post(f"{API}/v1/predict", json=body, headers=HEADERS, timeout=60)
        r.raise_for_status()
        j = r.json()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    else:
        st.metric("Delay probability", f"{j['delay_probability']:.0%}")
        msg = {"low": "Unlikely to be delayed", "elevated": "Possible delay", "high": "Likely to be delayed"}[j["risk"]]
        {"low": st.success, "elevated": st.warning, "high": st.error}[j["risk"]](msg)
        st.caption(
            f"model v{j['model_version']} · decision threshold {j['threshold']:.0%}"
            + ("" if j["known_route"] else " · new route for this model, so the estimate is less certain")
        )