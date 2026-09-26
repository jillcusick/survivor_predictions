# load raw data and split into chronological season groupings for train/test 

import pandas as pd 

def load_and_clean_data(file_path):
    """ Loads raw Survivor cast data, merges sources, and limits to data to use for modeling. """

    ### load and merge data
    
    # load raw data on castaway overview and results
    castaways = pd.read_excel(file_path, sheet_name="Castaways")
    
    # Keep variables of interest for modeling
    castaways_vars_keep = ['version', 'season', 'full_name', 'castaway_id', 'age', 'city', 'state', 'order', 'place', 'jury', 'finalist', 'winner']
    
    # load raw data on castaway details
    details = pd.read_excel(file_path, sheet_name="Castaway Details")
    
    # keep variables of interest for modeling
    details_vars_keep = ['castaway_id', 'gender', 'occupation', 'bipoc']
    
    # merge castaway data sources
    df = pd.merge(castaways[castaways_vars_keep], details[details_vars_keep], on='castaway_id', how='left')
    
    ### limit dataset to US seasons of the show
    df = df[df['version'] == 'US']
    
    return df

    
    
    
    
    
    