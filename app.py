"""Shared CSC 8830 entry point. From the repository root: streamlit run app.py.

Keep this filename for the existing deployment. Assignment implementations and
assets stay in their own packages; only the selected page runs on each request.
"""
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent


def main():
    st.set_page_config(page_title='CSC 8830 Computer Vision', layout='wide')
    page = st.navigation([
        st.Page(BASE_DIR / 'home.py', title='Home', default=True),
        st.Page(BASE_DIR / 'module2' / 'page.py', title='Module 2', url_path='module-2'),
        st.Page(BASE_DIR / 'module3' / 'page.py', title='Module 3', url_path='module-3'),
    ])
    # Streamlit removes widgets from inactive pages. Clear their associated
    # computed results too, so returning cannot reuse a stale upload/click event.
    # Ordinary widget reruns on the same page preserve the current selection.
    if st.session_state.get('navigation.active_page') != page.url_path:
        for key in list(st.session_state):
            if key.startswith(('module2.', 'module3.')):
                del st.session_state[key]
        st.session_state['navigation.active_page'] = page.url_path
    st.sidebar.caption('Changing pages starts a new selection. Download results before leaving a module.')
    page.run()


if __name__ == '__main__':
    main()
