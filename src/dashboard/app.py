import streamlit as st

st.set_page_config(
    page_title="RAG Query Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit header/footer style elements to give a native app feel
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# React URL input in the sidebar for configuration
react_url = st.sidebar.text_input("Vite Dev Server URL", value="http://localhost:5173")

# Full-bleed iframe embedding the React app
st.markdown(
    f'<iframe src="{react_url}" style="width:100%; height:calc(100vh - 100px); border:none; margin:0; padding:0;"></iframe>',
    unsafe_allow_html=True
)
