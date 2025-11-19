import streamlit as st
from app_home import run_home
from app_login import run_login

# Set Streamlit page config
st.set_page_config(page_title="vDPO Login Portal", layout="centered")

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Routing
if st.session_state.authenticated:
    run_home()
else:
    run_login()
