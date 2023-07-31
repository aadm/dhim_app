# dhimatrix_app
#
# $ streamlit run dhim_app.py
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from dhim_funcs import *

#==========================================================
# initialize app

st.set_page_config(page_title='DHIm app', layout="wide")
     
#==========================================================
# initialize sidebar

grade_options = ['Random', 'Predefined', 'Best', 'Worst']
pos_options = ['Random', 'Predefined']

with st.sidebar:
    st.write('_Predefined_ fills in probabilities and grades with default values and allows the user to modify values.')
    sel_grade_opt = st.selectbox('Grades', grade_options, index=1)
    # sel_pos_opt = st.selectbox('POS', pos_options, index=1)

if sel_grade_opt == 'Random':
    g_presel = np.random.choice(grade_types, size=24)
elif sel_grade_opt == 'Predefined':
    g_presel = ['A','B','C','A','D','B','B','D','C','C','B','D','C','C','B','C','B','B','B','A','A','A','B','B']
elif sel_grade_opt == 'Best':
    g_presel = ['D']*24
elif sel_grade_opt == 'Worst':
    g_presel = ['A']*24

# if sel_pos_opt == 'Random':
#     p_presel = np.random.choice(np.arange(0.3,1.,.1), size=5)
# elif sel_pos_opt == 'Predefined':
#     p_presel = np.array([.9, .9, .7, .4, .5])

#==========================================================
# initialize widgets

grade_types = ['A', 'B', 'C', 'D']
weight_types = ['W1', 'W2', 'W3']

riskelem = ['RES', 'SEA', 'SRC', 'TRP', 'CHG']

# assign bmt names, ids, extended descriptions
dhim_names = {
    # BACKGROUND TAB
    'DHI_BACK_1': ['BUR', 'Burial depth'],
    'DHI_BACK_2': ['PORO', 'Porosity range'],
    'DHI_BACK_3': ['FLUID', 'HC type'],
    'DHI_BACK_4': ['AGE', 'Age of target'],
    'DHI_BACK_5': ['DHIP', 'DHI play'],
    'DHI_BACK_6': ['SFACIES', 'Seismic facies'],
    'DHI_BACK_7': ['PITFALL', 'Possible pitfalls'],
    # DATA QUALITY TAB
    'DHI_DQ_1': ['SEISD', 'Seismic database'],
    'DHI_DQ_2': ['SEISQ', 'Seismic data quality'],
    'DHI_DQ_3': ['RPM', 'Rock physics model'],
    'DHI_DQ_4': ['CLASS', 'AVO class'],
    # AMPLITUDE ANOMALY TAB
    'DHI_AA_1': ['AMPC', 'Amplitude change'],
    'DHI_AA_2': ['ACONS', 'Amplitude consistency'],
    'DHI_AA_3': ['MODM', 'Seismic signature'],
    'DHI_AA_4': ['STRFIT', 'Amplitude structural fit'],
    # AVO TAB
    'DHI_AVO_1': ['AVOM', 'AVO analysis method'],
    'DHI_AVO_2': ['AVOA', 'AVO change & consistency'],
    'DHI_AVO_3': ['AVOVM', 'AVO signature'],
    'DHI_AVO_4': ['AVOFIT', 'AVO structural fit'],
    # OTHER DHIS TAB
    'DHI_OTHER_1': ['FSPOT', 'Flat spot'],
    'DHI_OTHER_2': ['PHASE', 'Phase change'],
    'DHI_OTHER_3': ['LOWF', 'Low frequency anomaly'],
    'DHI_OTHER_4': ['PULLD', 'Pull-down'],
    'DHI_OTHER_5': ['CHIM', 'Gas chimneys'],
}
    

# POS widgets
pos_widgets = st.columns(5, gap='medium')

tmp = np.full(5, np.nan)
opt0 = dict(value=1.0, min_value=0.0, max_value=1.0, step=0.05, format='%.2f')
for i, val in enumerate(pos_widgets):
    with val:
        tmp[i] = st.number_input(riskelem[i], **opt0)
pos_input = tmp

st.divider()

# DHI Matrix characteristics widgets

grade_widgets = st.columns(3, gap='medium')
tmp = np.full(len(g_presel), 'X')

blocks = ['**Background and Data**', '**Full stack and AVO**', '**Other DHIs**']
rr = [[0,11], [11,19], [19,24]]
dck = list(dhim_names.keys())

c = 0 
opt1 = dict(options=grade_types, horizontal=True)
for i, colg in enumerate(grade_widgets):
    with colg:
        subset = dck[rr[i][0]:rr[i][1]]
        st.write(blocks[i])
        for j, val in enumerate(subset):
            id = subset[j]    
            name = dhim_names[id][0]    
            tmp[c] = st.radio(name, index=grade_types.index(g_presel[c]), **opt1)
            c += 1

grades = tmp

st.divider()

#==========================================================
# run calculations

# update dhi matrix DataFrame with selected grades
dm = initialize_dhimatrix()
dm_updated, pdhi_idx, pitf1_idx, pitf2_idx, pitf_idx = update_dhi_matrix(dm, grades)

# calculate final POS
posg, pos, prb = update_pos(pos_input, pdhi_idx, pitf_idx)

#==========================================================
# summary tabs

# define prior and plot bayes posterior for all 5 risk model elements
prior = np.linspace(0, 1.)
tmp = np.zeros((prior.size, 5+1))
tmp[:, 0] = prior
for i in range(5):
    tmp[:, i+1] = posterior_probability(prior, prb['sens'].iloc[i], prb['spec'].iloc[i])

col_names = list(prb.index)
col_names.insert(0, 'PRIOR')
df = pd.DataFrame(tmp, columns = col_names)
df0 = df.melt('PRIOR', var_name='Risk Model Element', value_name='POSTERIOR')

# assemble input POS into a dataframe for plotting
points = pd.DataFrame.from_dict(dict(zip(prb.index, prb.posg)), orient='index', columns=['posg'])
points['pos']  = prb.pos
points = points.reset_index()

# hard-code upper and lower bounds for this version of the DHI Matrix
bayes_upper_bound = posterior_probability(prior, 0.804, 0.897)
bayes_lower_bound = posterior_probability(prior, 0.731, 0.275)

bounds = pd.DataFrame.from_dict({
    'PRIOR': prior,
    'Lower': bayes_lower_bound,
    'Upper':bayes_upper_bound})

# define list of 5 colors
colrs = alt.Scale(range=['olive', 'green', 'darkgrey','red', 'magenta'])

# setup altair charts
c1 = alt.Chart(df0).mark_line().encode(
    x='PRIOR:Q',
    y='POSTERIOR:Q',
    color=alt.Color('Risk Model Element:N', scale=colrs)
)

c2 = alt.Chart(points).mark_point().encode(
    x='posg:Q',
    y='pos:Q',
    color=alt.Color('index:N', scale=colrs)
)

opt = dict(filled=True, opacity=0.5, color='black')
c3 = alt.Chart(bounds).mark_point(**opt).encode(
    x='PRIOR:Q',
    y='Upper:Q')

c4 = alt.Chart(bounds).mark_point(**opt).encode(
    x='PRIOR:Q',
    y='Lower:Q')

chart = c1+c2+c3+c4

# setup tabs
tab1, tab2, = st.tabs(['output', 'plot'])

with tab1:
    col1, col2 = st.columns(2, gap='large')
    with col1:
        st.write('### POSg: :red[{:.2f}] // POS: :red[{:.2f}]'.format(posg, pos))
        st.write('Primary DHI Index: :red[{:.2f}]'.format(pdhi_idx))
        txt = 'Pitfall Index 1: :red[{:.2f}] // Index 2: :red[{:.2f}]'
        st.write(txt.format(pitf1_idx, pitf2_idx))
        st.write('Pitfall Index avg: :red[{:.2f}]'.format(pitf_idx))
    with col2:
        st.dataframe(prb, use_container_width=True)
    st.dataframe(dm_updated, use_container_width=True, hide_index=True, height=800)

with tab2:
    st.altair_chart(chart, use_container_width=True, theme="streamlit")
