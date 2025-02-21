import streamlit as st
import data_reduce as dr
import pandas as pd
import numpy as np

# Page setup
st.set_page_config("Data Reduction App", layout="wide", initial_sidebar_state="collapsed")
st.title("Data Reduction App")
st.markdown(f'_v{dr.__version__}_ | <a href="https://github.com/kstone40/data-reduce"><img src="https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png" alt="GitHub Repository" width="25"></a>', unsafe_allow_html=True)

# Set up sidebar
st.sidebar.title("Options")
st.sidebar.markdown("Select the options below:")

# Select algorithm
option = st.sidebar.selectbox("Choose a reduction method:", options=('Visvalingam-Whyatt', 'Downsampling'),
                              index=0)
                              
reducer_dict = {
    dr.VWReducer.type: dr.VWReducer,
    dr.DownSampler.type: dr.DownSampler
}
reducer = reducer_dict[option]()
st.write(f"Selected reduction method: {reducer.type}")

# Data input
if 'df_data' not in st.session_state:
    st.session_state.df_data = pd.DataFrame()

app_cols = st.columns(2)
with app_cols[0]:
    with st.expander("Data Input", expanded=True):
        
        upload_file = st.file_uploader("Upload a CSV file", type=['csv'])

        if upload_file:
            st.session_state.df_data = pd.read_csv(upload_file)
            if len(st.session_state.df_data.columns) != 2:
                st.error("Uploaded CSV must have exactly two columns.")
                st.session_state.df_data = pd.DataFrame()
            else:
                try:
                    st.session_state.df_data = st.session_state.df_data.apply(pd.to_numeric)
                except ValueError:
                    st.error("All data in the CSV must be numeric.")
                    st.session_state.df_data = pd.DataFrame()
                    
        button_cols = st.columns(3)
            
        with button_cols[0]:
            random_gen = st.button("Generate Random Data", type="primary")
        
        with button_cols[1]:
            clear_data = st.button("Clear Data", type="secondary")
            
        with button_cols[2]:
            pass
        
        if random_gen:
            m_points = int(1e3)
            x1 = np.linspace(-20, 20, m_points)
            
            # Randomly generate periodic and polynomial data
            sin_scale = np.random.uniform(-2, 2)
            sin_period = np.random.uniform(1, 3)
            sin_shift = np.random.uniform(-5, 5)
            cos_scale = np.random.uniform(-5, 5)
            cos_period = np.random.uniform(1, 3)
            cos_shift = np.random.uniform(-5, 5)
            linear_scale = np.random.normal(0, 1)
            quad_scale = np.random.normal(0, 0.05)
            error = np.random.normal(1, 0.05, m_points)
            
            x2 = (x1*sin_scale*np.sin((x1-sin_shift)/sin_period) \
                 + cos_scale*np.cos((x1-cos_shift)/cos_period) \
                 + x1*linear_scale + (x1**2)*quad_scale) * error
            
            st.session_state.df_data = pd.DataFrame({'X1': x1, 'X2': x2})

        if clear_data:
            st.session_state.df_data = pd.DataFrame(columns=['X1', 'X2'])

        st.data_editor(st.session_state.df_data, num_rows='dynamic', use_container_width=True)

with app_cols[1]:
    with st.expander("Data Reduction", expanded=True):
        if not st.session_state.df_data.empty:
            n_target = st.number_input("Target number of points", min_value=3, max_value=len(st.session_state.df_data),
                                       value=st.session_state.df_data.shape[0]//2, step=st.session_state.df_data.shape[0]//100)
            
            x = st.session_state.df_data.values
            
            with st.spinner("Reducing..."):
                if option == 'Visvalingam-Whyatt':
                    importances = st.cache_data(reducer.all_importances)(x[:, 0], x[:, 1])
                    x_culled = reducer.reduce(x, n_target, importances)        
                else:
                    x_culled = reducer.reduce(x, n_target)
            
            with st.spinner("Plotting..."):
                fig = dr.show_reduction(x, x_culled)
                fig.update_layout(height=445)
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig)
                
            reduced_data = pd.DataFrame(x_culled, columns=['X1', 'X2'])
            csv = reduced_data.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Download Reduced Data",
                data=csv,
                file_name='reduced_data.csv',
                mime='text/csv',
            )
                
        else:
            st.warning("Please input some data first.")