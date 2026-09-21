"""Assignment home page. Launch the shared app with: streamlit run app.py."""
import streamlit as st

st.title('CSC 8830 Computer Vision')
st.write('Jieming Li · Assignment demonstrations')
st.markdown('Choose an assignment below or use the sidebar. Both modules run in this application.')
book, blur = st.columns(2)
with book:
    with st.container(border=True):
        st.subheader('Module 2 · Book measurement')
        st.write('Estimate a front-facing book cover’s width and height from a calibrated phone image, '
                 'the camera-to-book distance, and four selected corners. View the saved evaluation '
                 'and download the measurement.')
        st.page_link('module2/page.py', label='Open Module 2')
with blur:
    with st.container(border=True):
        st.subheader('Module 3 · Image blurring')
        st.write('Compare explicit spatial convolution with Fourier-domain filtering. Choose a Box '
                 'or Gaussian kernel, inspect numerical agreement and difference maps, and download '
                 'the outputs.')
        st.page_link('module3/page.py', label='Open Module 3')
st.info('Switching pages starts a fresh interactive selection. Download results before leaving a module.')
