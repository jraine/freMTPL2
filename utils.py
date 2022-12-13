import pandas as pd
import numpy as np
import arff


#Onehot for categorical, scaler interesting for continuous floats, with decision tree likely only necessary on target
from sklearn.preprocessing import OneHotEncoder,StandardScaler #RobustScaler might be intersting but many zero values, could end up with mean, std of zeros for many observables


def load_arff_pd(filepath):#, indexvar='IDpol'):
    '''Function to load an arff file and return a pandas DataFrame
    Input: 
      filepath: path to file
    Returns:
      Pandas DataFrame
    '''
    #open arff file, using liac-arff, not familiar with format
    #There must be better loading tools or libraries compatible directly with this format
    #However I am unfamiliar with it
    data = arff.load(open(filepath,'r'))
    #convert to pandas dataframe
    cols = [x[0] for x in data['attributes']]
    df = pd.DataFrame(data=data['data'],columns=cols)
    #64->32bit precision, turn objects back into categories

    for col,dtype in zip(df.columns,df.dtypes):
        if dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif dtype == 'object':
            df[col] = df[col].astype('category')
    return df


def load_merged_dataframe(freqfile,sevfile):
    '''Function to return dataframe with columns merged.
    Input:
      freqfile: File with policy information
      sevfile: File with claims information
    Return:#
      combined dataframe'''

    fq_df = load_arff_pd(freqfile)
    sv_df = load_arff_pd(sevfile)

    # first sum claim amounts to be one per ID
    sv_df = sv_df.groupby('IDpol').sum()
    df = fq_df.join(sv_df,on='IDpol').sort_index()
    # make sure policies with no claims get claimamount set to zero
    df.ClaimAmount = df.ClaimAmount.fillna(0.0)

    return df

def cat_to_onehot(data, columns=None):
    '''
    Function to return data with categorical features converted to one hot vectors
    Input:
      data: input dataframe
      columns: columns to return, otherwise all categorical returned onehot
    Output:
      dataset with onehot encodings'''

    encoder = OneHotEncoder(handle_unknown='ignore')
    if columns is None:
        columns = [c for c,d in zip(data.columns,data.dtypes) if d != 'float32']
    elif type(columns) not in [list,tuple]:
        columns = [columns]
    return encoder.fit_transform(data[columns]).toarray()

def mean_std_floats(data, columns=None, transformer=None):
    '''
    Function to return data with float features scaled with Standard Scaler
    Input:
      data: input dataframe
      columns: columns to return, otherwise all categorical returned onehot
    Output:
      transformed floats
      transformer'''
    if columns is None:
        columns = [c for c,d in zip(data.columns,data.dtypes) if d == 'float32']
    elif type(columns) not in [list,tuple]:
        columns = [columns]
    if transformer is None:
        transformer = StandardScaler()
        data = transformer.fit_transform(data[columns])
    else:
        data = transformer.transform(data[columns])

    return data, transformer
  
def inverse_transform_floats(data, transformer):
    '''
    Function to inverse transformation of float data
    Input:
      data: input numpy array
      transformer: transformer to invert
    Output:
      untransformed floats'''
    data = transformer.inverse_transform(data)
    return data