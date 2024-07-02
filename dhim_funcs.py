#----------------------------------------------------------
# aadm 2022-2023
# last update: 2023-06-22
#----------------------------------------------------------

import numpy as np
import pandas as pd

def marginal_probability(prior, sens, spec):
    '''
    Calculate marginal probability of A (test, presence of DHIs)
    i.e. probability of having DHIs at all, wether or not hydrocarbons are present
    '''
    return sens*prior + (1-spec)*(1-prior) # P(B)


def posterior_probability(prior, sens, spec):
    '''
    Calculate posterior probability P(H|A)
    i.e. prob of event H (presence of hydrocarbon) being true
    if A (test, presence of DHIs) is true
    '''
    return (sens*prior) / marginal_probability(prior, sens, spec)


# transfer function to convert Primary DHI Index to Spec=Sens
def dhip_to_spec(x):
    return -.697*x**2 + 1.584*x + 0.053

# squashing function to create Sens from Spec
def spec_to_sens(s, p):
    return (p-.05) * (((1-s)-s) / (.82-.05)) + s

# general definitions
# define columns
grade_types = ['A', 'B', 'C', 'D']
weight_types = ['W1', 'W2', 'W3']

def initialize_dhimatrix():
    # INITIALIZE DATAFRAME
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
    
    # assign scores
    # i.e. the numerical equivalent of grades A, B, C, D for each characteristic
    # TAKEN FROM DM3-PAPA_v22-00.xlsx
    scores = np.array([
        [4, 3, 2, 1],
        [4, 3, 2, 1],
        [4, 3, 2, 1],
        [4, 3, 2, 1],
        [2, 5, 7, 10],
        [5, 7, 9, 10],
        [3, 5, 9, 10],
        [6, 7, 8, 10],
        [2, 5, 8, 10],
        [0, 6, 8, 10],
        [0, 8, 10, 9],
        [0, 5, 8, 10],
        [0, 3, 7, 10],
        [0, 4, 7, 10],
        [0, 4, 7, 10],
        [0, 4, 7, 10],
        [0, 2, 7, 10],
        [0, 2, 7, 10],
        [0, 4, 6, 10],
        [2, 5, 7, 10],
        [4, 5, 7, 10],
        [5, 6, 8, 10],
        [5, 6, 8, 10],
        [2, 5, 7, 10],
    ])

    # assign class weights
    # i.e. modifiers for scores on the basis of selected AVO class
    # TAKEN FROM DM3-PAPA_v22-00.xlsx
    class_weights = np.array([
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [0.7,1.0,1.2],
        [1.0,1.4,1.2],
        [1.4,1.7,1.9],
        [1.3,1.2,1.1],
        [0.7,1.4,1.0],
        [1.0,1.9,1.4],
        [1.0,1.2,1.4],
        [1.4,1.9,1.0],
        [1.4,0.5,1.9],
        [1.3,0.7,1.0],
        [1.9,1.4,1.9],
        [1.0,1.4,1.2],
        [1.3,1.0,1.0],
        [1.0,1.0,0.7],
        [1.0,1.0,1.0],
        [0.1,0.5,0.5],
        [0.1,0.5,0.5],
        [0.1,0.5,0.5],
        [0.1,0.4,0.7],
        [1.0,1.0,1.0],
    ])
    
    # assemble everything into a DataFrame
    tmp1 = pd.DataFrame.from_dict(dhim_names, orient='index', columns=['ID', 'Description'])

    # extract number of characteristics to grade: 24
    ncar = tmp1.shape[0]

    # make temporary matrix with all other components of the "engine"
    cols = ['A', 'B', 'C', 'D', 'W1', 'W2', 'W3', 'CF', 'MAXWF', 'GRADE', 'POWER']
    nans = np.full(shape=(ncar, len(cols)),fill_value=np.nan)
    tmp2 = pd.DataFrame(nans, index=tmp1.index, columns=cols)

    # combine the two DataFrames 
    dm = pd.concat([tmp1, tmp2], axis=1)

    # fill in grade "scores", i.e. associate numerical values (scores)
    # to selectable grades (A, B, C, D)
    # fill in weight "scores"
    for i, nn in enumerate(dm['ID']):
        dm.loc[dm['ID']==nn, grade_types] = scores[i, :]
        dm.loc[dm['ID']==nn, weight_types] = class_weights[i, :]
    
    return dm

def update_dhi_matrix(dhimatrix, grades):    
 
    dm = dhimatrix.copy()
 
    # define list of IDs
    # all_ids = all IDs for entire DHI Matrix
    # power_ids = only the IDs where POWER is calculated
    # backgr_ids = background IDs
    # dataqu_ids = data quality IDs
    class_ids = ['BUR', 'PORO', 'FLUID','AGE']
    all_ids = dm['ID'].to_numpy()
    power_ids = dm.loc[~dm['ID'].isin(class_ids), 'ID'].to_numpy()
    backgr_ids = dm.loc[dm.index.str.startswith('DHI_BACK'), 'ID'].to_numpy()
    dataqu_ids = dm.loc[dm.index.str.startswith('DHI_DQ'), 'ID'].to_numpy()
    
    # primary_ids = to calculate primary DHI index
    incl_backgr = dm['ID'].isin(backgr_ids)
    incl_dataqu = dm['ID'].isin(dataqu_ids)
    tmp = dm.loc[~(incl_backgr.values | incl_dataqu.values), 'ID'].to_numpy()
    index_AVOM = np.argwhere(tmp=='AVOM')
    primary_ids = np.delete(tmp, index_AVOM)

    #>>> USER INPUT: grades
    dm['GRADE'] = grades

    # convert grades to numbers (POWER) for first 4 background characteristics
    for nn in class_ids:
        assigned_grade = dm.loc[dm['ID']==nn, 'GRADE']
        score_grade = dm.loc[dm['ID']==nn, assigned_grade].to_numpy().flatten()
        dm.loc[dm['ID']==nn, 'POWER' ] = score_grade

    # COMPCL: computed avoclass from background characteristics 
    # to be used when avo class is unknown (grade A in DHI_DQ_4 or 'CLASS')
    coeffs = [0.32, 0.25, 0.20, 0.05]
    class_scores = dm.loc[dm['ID'].isin(class_ids), 'POWER'].to_numpy()
    computed_class = np.round(class_scores * coeffs)
    computed_class = int(np.round(np.sum(coeffs * class_scores)))

    # get avo class from DHI_DQ_4 ('CLASS')
    assigned_avoclass = dm.loc[dm['ID']=='CLASS', 'GRADE']
    if 'A' in assigned_avoclass.values:
        avoclass = computed_class
    elif 'B' in assigned_avoclass.values:
        avoclass = 1
    elif 'C' in assigned_avoclass.values:
        avoclass = 2
    elif 'D' in assigned_avoclass.values:
        avoclass = 3

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # deactivate a few attributes on the basis of context
    # i.e. a few conditions that make certain attributes not reliable

    # assign context flags
    dm['CF'] = True

    # if SFACIES=A (seismic facies is chaotic) then ACONS and MODM=0
    condition = dm.loc[dm['ID']=='SFACIES', 'GRADE'].isin(['A'])
    dm.loc[dm['ID']=='ACONS', 'CF'] = np.where(condition, False, True)
    dm.loc[dm['ID']=='MODM', 'CF'] = np.where(condition, False, True)

    # if AVOM=A (no avo analysis) then all other AVO charact=0
    condition = dm.loc[dm['ID']=='AVOM', 'GRADE'].isin(['A'])
    dm.loc[dm['ID']=='AVOA', 'CF'] = np.where(condition, False, True)
    dm.loc[dm['ID']=='AVOVM', 'CF'] = np.where(condition, False, True)
    dm.loc[dm['ID']=='AVOFIT', 'CF'] = np.where(condition, False, True)

    # if SFACIES=D (seismic facies is single loop reflection) then FSPOT=0
    condition = dm.loc[dm['ID']=='SFACIES', 'GRADE'].isin(['D'])
    dm.loc[dm['ID']=='FSPOT', 'CF'] = np.where(condition, False, True)

    # if FLUID=C, D (oil) or PORO=D (low porosity) then PULLD=0
    fluid_is_oil = dm.loc[dm['ID']=='FLUID', 'GRADE'].isin(['C', 'D']) 
    low_poro = dm.loc[dm['ID']=='PORO', 'GRADE'].isin(['D']) 
    condition = fluid_is_oil.values or low_poro.values
    dm.loc[dm['ID']=='PULLD', 'CF'] = np.where(condition, False, True)

    # if FSPOT=A (flat spot expected but not found) then CHIM=0
    condition = dm.loc[dm['ID']=='FSPOT', 'GRADE'].isin(['A'])
    dm.loc[dm['ID']=='CHIM', 'CF'] = np.where(condition, False, True)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # calculate MAXWF and make it 0 if context flag is False
    dm['MAXWF' ] = dm[weight_types].max(axis=1) * dm['CF']

    # calculate POWER = scores * avo weight * context flags
    # select all IDs except the first 4 background characteristics
    for nn in power_ids:
        selec_id = dm['ID']==nn
        assigned_grade = dm.loc[selec_id, 'GRADE']
        score_grade = dm.loc[selec_id, assigned_grade].to_numpy().flatten()   
        weight = dm.loc[selec_id, 'W'+str(avoclass)].to_numpy().flatten()
        context_flag = dm.loc[selec_id, 'CF'].to_numpy().flatten()   
        dm.loc[selec_id, 'POWER' ] = score_grade * weight * context_flag /10

    # PRIMARY DHI INDEX
    above = dm.loc[dm['ID'].isin(primary_ids), 'POWER'].sum()
    below = dm.loc[dm['ID'].isin(primary_ids), 'MAXWF'].sum()
    primary_dhi_index = above / below

    # EXTENDED DHI INDEX
    above = dm.loc[dm['ID'].isin(power_ids), 'POWER'].sum()
    below = dm.loc[dm['ID'].isin(power_ids), 'MAXWF'].sum()
    extended_dhi_index = above / below

    pitfall_1_ids = ['DHIP', 'SFACIES', 'SEISQ', 'CLASS', 'AVOM']
    above = dm.loc[dm['ID'].isin(pitfall_1_ids), 'POWER'].sum()
    below = dm.loc[dm['ID'].isin(pitfall_1_ids), 'MAXWF'].sum()
    pitfall_1_index = 1-(above / below)

    pitfall_2_ids = ['PITFALL', 'SEISD', 'RPM', 'AVOM']
    above = dm.loc[dm['ID'].isin(pitfall_2_ids), 'POWER'].sum()
    below = dm.loc[dm['ID'].isin(pitfall_2_ids), 'MAXWF'].sum()
    pitfall_2_index = 1-(above / below)

    pitfall_index = np.mean([pitfall_1_index, pitfall_2_index])

    return dm, primary_dhi_index, pitfall_1_index, pitfall_2_index, pitfall_index

def update_pos(pos_input, primary_dhi_index, pitfall_index, fancy=True):

    # build engine dataframe "prb"
    riskelem = ['RES', 'SEA', 'SRC', 'TRP', 'CHG']
    
    #>>> USER INPUT: probabilities risk elements
    pos_elem = dict(zip(riskelem, pos_input))
    prb = pd.DataFrame.from_dict(pos_elem, orient='index', columns=['posg'])

    # add specificity
    prb['spec'] = dhip_to_spec(primary_dhi_index)

    # add pitfall index modified for risk elements
    if fancy:
        coeff_pitfall_index = [.7, .5, 1, .3, 1]
        prb['pitf'] = pitfall_index**coeff_pitfall_index 
    else:
        prb['pitf'] = pitfall_index
 
    # add sensitivity
    prb['sens'] = spec_to_sens(prb['spec'], prb['pitf'])

    # calculate POS with Bayes on risk elements
    prb['pos'] = posterior_probability(prb['posg'], prb['sens'], prb['spec'])

    # calculate POSg and updated POS
    posg = prb['posg'].product()
    pos = prb['pos'].product()

    return posg, pos, prb

