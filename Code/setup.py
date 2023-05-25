
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import pickle
import math
import shutil
import torch
from torch.utils.data import Dataset
import datetime as dt
import torch.nn as nn
import copy
from IPython.display import clear_output
from collections import defaultdict
import random
import tensorflow as tf

# n_ins = 4
# n_dbs = 4
# database_path = r'd:\!Study\Python\test data' #Перенести в директорию проекта
# in_list = 'initial files.txt'
# new_list = 'current files.txt'
# good_columns = ['Ns', 'time/s', "Ewe/V", "<I>/mA", 'cycle number', '(Q-Qo)/mA.h', 'Capacity/mA.h']
# database_path_short = r'd:\!Study\Python\test data short' #Перенести в директорию проекта

def get_capacity(df):
    """Get dataframe of a cycle, and returns discharge capacity, from two sources (return tuple of 2 values)"""
    difference = df[df['<I>/mA']<-74].iloc[0] - df[df['<I>/mA']<-74].iloc[-1]
    Q, C = difference[['(Q-Qo)/mA.h', 'Capacity/mA.h']]
    return Q, C

def extract_params(df):
    """extract_params Takes from dataframe Eup, Edown, current and initial guess for capacity. For better understanding see Thevenin_model class

    Args:
        df (DataFrame): one cycle of GCPL

    Returns:
        tuple of floats Eup, Edown, current, Qinit: parameters needed for model creation and initialization of first guesses
    """
    Eup = df[df['<I>/mA']<-74].iloc[0]['Ewe/V']
    Edown = df[df['<I>/mA']<-74].iloc[-1]['Ewe/V']
    current = df[df['<I>/mA']<-74]['<I>/mA'].mean()
    Qinit, _ = get_capacity(df)
    return Eup, Edown, current, Qinit

def prepareData(df):
    """ Extract from all cycle discharging cycle, and then return three 
    numpy arrays of E, time and Q."""
    E = df[df['<I>/mA'] < -74]['Ewe/V'].to_numpy(copy=True)
    time = df[df['<I>/mA'] < -74]['time/s'].to_numpy(copy=True) 
    time = time - time.min()
    Q = df[df['<I>/mA'] < -74]['Capacity/mA.h']
    return E, time, Q

def gcpl_select_cycle(df,
                      cycles: list,
                      x='time/s',
                      y='Q charge/discharge/mA.h'):
    """Gets dataframe from mpt file, and return dataframe with selected cycle numbers and columns as x and y"""
    df_selected = df.loc[df['cycle number'].isin(cycles), [x, y]]
    return df_selected


def gcpl_view(df,
              step=50,
              rows=4,
              cols=5,
              x='time/s',
              y='Q charge/discharge/mA.h',
              start_cycle=0):
    """Get dataframe made from mpt file and draws scatters in multiple plots. 
    You can select start cycle , step - number of cycles in each plot, what 
    columns to draw (x an y as column names in dataframe), how many plots there 
    will be (nplots = ncols*nrows). """
    cycles = np.arange(start_cycle, step*rows*cols+start_cycle)
    cycles_draw = cycles.reshape(rows,cols,-1)
    index = list(product(range(rows), range(cols)))
    fig, ax = plt.subplots(nrows=rows, ncols=cols,figsize = (16,10), sharey=True)
    fig.suptitle(f'{y} vs {x}')
    if max(cycles)< df['cycle number'].max() or min(cycles)> df['cycle number'].min():
        print('Warning! Not all cycles are displayed!')
    for a, b in index:
        df_draw = gcpl_select_cycle(df, cycles_draw[a,b], x=x, y=y)
        ax[a,b].scatter(df_draw[x],df_draw[y],1)
        min_, max_ = cycles_draw[a,b].min(), cycles_draw[a,b].max()
        ax[a,b].title.set_text(f'{min_} - {max_}')
        ax[a,b].set_xticklabels([])
        

def gcpl_abcent_cycles(df):
    """Gets dataframe of gcpl cycling and returns list of numbers of what cycles are missed from experiment."""
    max_ = df['cycle number'].max()
    min_ = df['cycle number'].min()
    all_cycles = np.arange(min_, max_)
    abcent = np.setdiff1d(all_cycles, df['cycle number'])
    print(f'cycles from {min_} to {max_}')
    return abcent


def list_files(filedir, fmt='mpt'):
    """You give a directory, and function returns list of all files (filepaths) with selected format fmt"""
    tree = os.walk(filedir)
    all_files = []
    for path, dirs, files in tree:
        for file in files:
            if file.split('.')[-1] == fmt:
                all_files.append(path+'\\'+file)
    return all_files


def make_dataframe_mpt(filepath):
    """Take file (as filepath of GCPL experiment in mpt format) and return dict with dataframe with experimental data ("data"), header 'info' and filepath 'path'"""
    file_dict = {}
    file_dict['path'] = filepath
    with open(filepath, 'r') as file:
        file.readline()
        n_header = int(file.readline().split()[-1])
        info = []
        for _ in range(n_header-3):
            info.append(file.readline())
        file_dict['info'] = info
        head = file.readline()
    file_data = pd.read_csv(filepath, sep='\t', encoding='ansi', header = n_header-1, skip_blank_lines=False)
    file_dict['data'] = file_data
    return file_dict


def save_list(lst, listname, path):
    listdir = path + '\\'+ listname
    with open(listdir, 'w') as f:
        f.writelines('\n'.join(lst))


def load_list(listname, path):
    listdir = path + '\\'+ listname
    with open(listdir) as f:
        in_filelist_ = f.read().split('\n')
    return in_filelist_


def make_datalist(filedir):
    """You give a directory, and function returns list of all folders with only files inside (filepaths)"""
    tree = os.walk(filedir)
    all_files = []
    for path, dirs, files in tree:
        if dirs == []:
            all_files.append(path)
    return all_files


# good_columns = ['Ns', 'time/s', "Ewe/V", "<I>/mA", 'cycle number', '(Q-Qo)/mA.h', 'Capacity/mA.h']


def repair_df(df, good_columns):
    """repair_df This function created to modify existing GCPL experiment from specific problems - point duplication, half-cycles and high/low potentials. 
    If cycle is not full, 
    Also deletes all columns except good_columns!
    returns repaired dataframe

    Args:
        df (Dataframe): Dataframe to be repaired
        good_columns (_type_): _description_

    Returns:
        _type_: _description_
    """
    copy_df = df[good_columns].copy(deep=True)
    cycles = copy_df['cycle number'].unique()
    drop_cycles = list(copy_df[((copy_df['Ewe/V']> 4.21) | (copy_df['Ewe/V']< 2.7))]['cycle number'].unique())
    for i in cycles:
        if len(copy_df[copy_df['cycle number'].isin([i])]['Ns'].unique()) != 2:
            drop_cycles.append(i)
    copy_df.drop(index=copy_df[copy_df['cycle number'].isin(drop_cycles)].index, inplace=True)
    for i in range(len(copy_df)):
        if i+1 >= len(copy_df):
            break
        while copy_df['time/s'].iloc[i] >= copy_df['time/s'].iloc[i+1]:
            # print(i,copy_df['time/s'].iloc[i], copy_df['time/s'].iloc[i+1])
            copy_df.drop(index = i+1, inplace = True)
            copy_df.reset_index(drop=True,inplace=True)
    return copy_df








def files_in_dir(filedir, fmt):
    return ['\"'+ filedir[0] +"\\" +file+'\"' for file in filedir[2] if file.split('.')[-1] == fmt]


class GCPL_exp:
    """This class is created for handling gcpl files, parsing it, saving to selected folder with known file design. Also it can load already saved data."""
    def __init__(self, path, raw=False, database_path = None, n_ins=None, n_dbs=None):
        """Loads GCPL experiment from folder 'path', or create GCPL experiment 
        structure if raw==True. For raw creation you give database_path as root 
        for new database, n_ins as depth of non-sense folders, n_dbs for depth 
        of database. For example,   
        d:\\!Science\\Nissan\\Analysis\\Electrochem\\Pouches\\G_series\\G01\\01 GCPL Aging\\030222 GCPL G01-G12 C2 500 cycles continued2_C01_fix.mpt - 
        consists of 6 needless folders d:\\!Science\\Nissan\\Analysis\\Electrochem\\Pouches and necessary folders G_series\\G01\\01 GCPL Aging\\030222 
        GCPL G01-G12 C2 500 cycles continued2_C01_fix.mpt. If folders less than n_dbs, folders named Unknown are nested at position[-2].Gool luck!"""
        
        self.filenames = ['header.txt', 'info.pkl', 'data.csv']
        self.file_keys = ['header', 'info', 'data']
        if raw == True:
            self.make_dataframe_mpt(path)
            self.create_database_entry(database_path, n_ins, n_dbs)
        else:
            self.load(path)
            self.path = path
        
        self.dataslice = self.data
        self.abcent_cycles()
        self.info['current dir'] = self.path
        self.n_cycles()
        self.length()
        
    def make_dataframe_mpt(self, path):
        """Take file (as filepath of GCPL experiment in mpt format) and return dict with dataframe
        with experimental data ("data"), header 'header' and filepath 'path'. Also it creates dict named 'info'"""
        self.in_path = path
        with open(self.in_path, 'r') as self.file:
            self.file.readline()
            n_header = int(self.file.readline().split()[-1])
            self.header = []
            for _ in range(n_header-3):
                self.header.append(self.file.readline())
        self.data = pd.read_csv(self.in_path, sep='\t', encoding='ansi', header = n_header-1, skip_blank_lines=False)
        if 'cycle number' in self.data.columns:
            self.data['cycle number']=self.data['cycle number'].astype(int)
        self.info = {'initial path': self.in_path}
        
    
    def create_database_entry(self, database_path, n_ins, n_dbs):
        """This function creates an entry of database at specific
        location database_path, with n_dbs depth and n_ins ignored
        root folders. in_path - filepath of initial file mpt. """
        nesting = self.in_path.split('\\')[n_ins:]
        
        while len(nesting) < n_dbs:
            nesting.insert(-1, 'Unknown')
        if len(nesting) > n_dbs:
            nesting = nesting[:n_dbs-1]+[nesting[-1]]
        self.info['nesting'] = nesting
        self.path = '\\'.join([database_path]+ nesting)
        self.save(path= self.path ,overwrite=True)
        
    
    def save(self,path, overwrite=False):
        """Saves current GCPL class data, for 3 files with each new method
        - text file for header, pickle dump for info, and csv for data."""
        if overwrite:
            self.path = path
        if not os.path.exists(path):
            os.makedirs(path)

        self.files= {i: self.make_fullpath(j, path) for i, j in zip(self.file_keys, self.filenames)}
        if overwrite:
            with open(self.files['header'], 'w') as f:
                f.writelines(self.header)
            with open(self.files['info'], 'wb') as f:
                pickle.dump(self.info, f)
            self.data.to_csv(self.files['data'], sep='\t',index=False)
            
        else:
            if os.path.exists(self.files['header']):
                print('Header already exist, not modified')
            else:
                with open(self.files['header'], 'w') as f:
                    f.writelines(self.header)
            if os.path.exists(self.files['info']):
                print('Info already exist, not modified')
            else:
                with open(self.files['info'], 'wb') as f:
                    pickle.dump(self.info, f)
            if os.path.exists(self.files['data']):
                print('Data already exist, not modified')
            else:
                self.data.to_csv(self.files['data'], sep='\t',index=False)
    
    def load(self, path):
        """Loads data, info and header from path directory.
        Rewrites existing data in class."""
        self.files = {i: self.make_fullpath(j, path) for i, j in zip(self.file_keys, self.filenames)}
        with open(self.files['header']) as f:
            self.header = f.readlines()
        with open(self.files['info'], 'rb') as f:
            self.info = pickle.load(f)
        self.data = pd.read_csv(self.files['data'], sep='\t', )

    def make_fullpath(self, filename, path):
        "Creates full paths from dirpath and filename"
        return path + '\\' + filename

    def select_cycles(self, *cycles, by_number=False, columns=None): 
        """Return data by selected cycles. If you select by_number=True, cycles are adressed directly. If no (by default) it returns cycles between 0 and len(gcpl) ignoring abcent cycles. Also you can select specific columns
        names that will be in slice.
        Also saves this slice to self.dataslice.
        columns should be list of column names! """
        if by_number:
            cycles = cycles
        else:
            cycles = [self.info['cycles'][i] for i in [*cycles]]
        
        if columns:
            df_selected = self.data.loc[self.data['cycle number'].isin(cycles), columns]
        else:
            df_selected = self.data.loc[self.data['cycle number'].isin(cycles), :]
        self.dataslice = df_selected
        return df_selected
    
    def abcent_cycles(self):
        """Creates in 'info' list of cycles that are missed from 
        experiment. Also creates min and max cycle, and total number of cycles"""
        self.info['max cycle'] = self.data['cycle number'].max()
        self.info['min cycle'] = self.data['cycle number'].min()
        if (self.info['max cycle'] <= self.info['min cycle']) or math.isnan(self.info['max cycle']) or math.isnan(self.info['min cycle'])  :
            self.info['abcent cycles'] = []
        else:
            all_cycles = np.arange(self.info['min cycle'], self.info['max cycle'])
            self.info['abcent cycles'] = np.setdiff1d(all_cycles, self.data['cycle number'])
        # print(f"cycles from {self.info['min cycle']} to {self.info['max cycle']}")
        
    def view(self,
          step=50,
          rows=4,
          cols=5,
          x='time/s',
          y='Q charge/discharge/mA.h',
          start_cycle=0):
        """Draws scatters in multiple plots. 
        You can select start cycle , step - number of cycles in each plot, what 
        columns to draw (x an y as column names in dataframe), how many plots there 
        will be (nplots = ncols*nrows). """
        cycles = np.arange(start_cycle, step*rows*cols+start_cycle)
        cycles_draw = cycles.reshape(rows,cols,-1)
        index = list(product(range(rows), range(cols)))
        fig, ax = plt.subplots(nrows=rows,
                               ncols=cols,
                               figsize=(16, 15),
                               sharey=True)
        fig.suptitle(f'{y} vs {x}')
        if max(cycles) < self.info['max cycle'] or min(cycles) > self.info['min cycle']:
            print('Warning! Not all cycles are displayed!')
        for a, b in index:
            df_draw = self.select_cycles(cycles_draw[a, b], columns=[x, y])
            ax[a, b].scatter(df_draw[x], df_draw[y], 1)
            min_, max_ = cycles_draw[a, b].min(), cycles_draw[a, b].max()
            ax[a, b].title.set_text(f'{min_} - {max_}')
            ax[a, b].set_xticklabels([])
    
    def n_cycles(self):
        """Saves in self.info['num cycles'] number of available cycles 
        and list of these cycles."""
        self.info['num cycles'] = self.data['cycle number'].nunique()
        self.info['cycles'] = self.data['cycle number'].unique()
        
    def length(self):
        """Add to self.info 'len' number of dots for dataframe"""
        self.info['len'] = self.data.shape[0]
    
    def __len__(self):
        return self.info['num cycles']
    
    def __getitem__(self, idx):
        return self.select_cycles(idx)
    
    def calculate_capacity(self, cycle):
        """calculate_capacity function returns capacity on charge and discharge at given cycle in experiment

        Args:
            cycle (int): cycle number to calculate capacity

        Returns:
            tuple of 2 floats: Charging capacity and discharging capacity.
        """
        df = self.select_cycles(cycle)
        Qcharge = df[df['Ns']==1]['Capacity/mA.h'].max()
        Qdischarge = df[df['Ns']==2]['Capacity/mA.h'].max()
        return Qcharge, Qdischarge
        
        
class GCPL_dataset(Dataset):
    """Dataset created for access to all data from cycling. This class is
    necessary for adressing every cycle through all experiments.
    It contains only header information like"""
    def __init__(self, database_list, GCPL_class = GCPL_exp):
        self.exp=GCPL_class
        self.files = database_list
        self.len = 0
        self.adress_list = []
        self.info = pd.DataFrame({
                                    'Pouch':[],
                                    'Experiment':[],
                                    'Filename':[],
                                    'Qc':[],
                                    'Qd':[], 
                                    'Qc_next': [],
                                    'Qd_next':[]
                                    })
        for file in tqdm(self.files):
            self.exp_cur = self.exp(file)
            self.len += len(self.exp_cur)
            self.adress_list += [(file, i) for i in range(len(self.exp_cur))]
            pouch = self.exp_cur.info['nesting'][1]
            experiment = self.exp_cur.info['nesting'][2]
            filename = self.exp_cur.info['nesting'][3]
            for i in range(len(self.exp_cur)):
                Qc, Qd = self.exp_cur.calculate_capacity(i)
                if i+1 < len(self.exp_cur):
                    Qc_next, Qd_next = self.exp_cur.calculate_capacity(i+1)
                else:
                    Qc_next, Qd_next = np.NaN, np.NaN

                    
                self.info.loc[len(self.info)] = [pouch,experiment, filename,  Qc, Qd,Qc_next, Qd_next] # type: ignore
        pouches = self.info['Pouch'].unique()
        self.pouches_capacity = {i:self.info[self.info['Pouch']==i]['Qd'].max() for i in pouches}
        self.info['SoH'] = self.info.apply(lambda x: x['Qd']/self.pouches_capacity[x['Pouch']]*100, axis=1)
        self.info['SoH_next'] = self.info.apply(lambda x: x['Qd_next']/self.pouches_capacity[x['Pouch']]*100, axis=1)


    def __len__(self):
        return self.len


    def __getitem__(self,idx, output=None):
        file, i = self.adress_list[idx]
        if file != self.exp_cur.path:
            self.exp_cur = self.exp(file)
        # if 'SoH' in output:
        #     return self.exp_cur[i], self.info.loc[i, 'SoH']
        return self.exp_cur[i], self.info.loc[idx]

class GCPL_dataset_preparator(Dataset):
    def __init__(self, GCPL_dataset, Preparator, savepath = None):
        self.data = []
        self.Preparator = Preparator
        self.path = savepath
        for i in range(len(GCPL_dataset)):
            result = Preparator(GCPL_dataset[i])
            if result[0]:
                self.data.append(result[1])
        median, std = self.median()
        self.del_outliers(median+std*3)
        if savepath:
            self.save(savepath)

    def median(self):
        mean = []
        for i in self.data:
            mean.append(len(i['E']))
        mean = np.array(mean)
        return np.median(mean), np.std(mean)
    

    def del_outliers(self, n):
        """n = max number of dots."""
        new_data = []
        for curr in self.data:
            if len(curr['E'])< n:
                new_data.append(curr)
        self.data = new_data


    def save(self, path=None, overwrite=False):
        if overwrite and os.path.exists(path):
            shutil.rmtree(path)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path+f'/data.pkl', 'wb') as f:
            pickle.dump(self.data, f) 

class GCPL_Preparator:
    """ This preparator modify all cycle values, standartize I to C-values, E to 0-1 for 2.75-4.2 range, capacity to C-values, Q 0-1 range, SoH to -1,1 range
    """
    def __init__(self, timedelta=60, capacity= 150, ):
        self.delta = dt.timedelta(seconds=timedelta)
        self.capacity = capacity


    def __call__(self, cycle):
        df, info = cycle
        soh = info['SoH']
        soh_n = info['SoH_next']
        if np.isnan(np.sum(soh)):
            return (False, False)
        if soh <5:
            return (False, False)
        curr = {}
        curr.update(info)
        curr['SoH'] = soh/50-1
        curr['SoH_next'] = soh_n/50-1
        curr['Cycle'] = int(df['cycle number'].unique())
        Q_start = df['(Q-Qo)/mA.h'].iloc[0]
        
        time = pd.DatetimeIndex(df['time/s'].apply(dt.datetime.fromtimestamp))
        df.set_index(time, inplace=True)
        df = df.resample(self.delta).mean()
        df.interpolate(method='linear', inplace=True)
        curr['E'] = (df['Ewe/V'].to_numpy(copy=True) - 2.75)/1.45
        curr['Q'] = (df['(Q-Qo)/mA.h'].to_numpy() - Q_start)/self.capacity
        curr['I'] = df['<I>/mA'].to_numpy(copy=True)/self.capacity
        curr['datetime'] = df.index.to_numpy(copy=True)
        return (True, curr)

class GCPL_dataset_resampled(Dataset):
    def __init__(self, GCPL_dataset, minLength = 0, maxLength= np.inf, sampling = 60, load=False):
        """sampling - number of seconds."""
        if load==False:
            self.data = []
            self.delta = dt.timedelta(seconds=sampling)
            for i in range(len(GCPL_dataset)):
                df, SoH, SoH_n = GCPL_dataset[i]
                if not ((df.shape[0] < maxLength) and (df.shape[0] > minLength)):
                    continue
                if np.isnan(np.sum(SoH)):
                    continue
                curr = {'SoH': SoH}
                curr['SoH_next'] = SoH_n
                time = pd.DatetimeIndex(df['time/s'].apply(dt.datetime.fromtimestamp))
                df.set_index(time, inplace=True)
                df = df.resample(self.delta).mean()
                df.interpolate(method='linear', inplace=True)
                curr['E'] = df['Ewe/V'].to_numpy()
                curr['I'] = df['<I>/mA'].to_numpy()
                curr['time'] = df['time/s'].to_numpy()
                curr['datetime'] = df.index.to_numpy()
                self.data.append(curr)
        else:
            self.load(GCPL_dataset)


    def save(self, path, overwrite=False):
        if overwrite and os.path.exists(path):
            shutil.rmtree(path)
        if not os.path.exists(path):
            os.makedirs(path)
        for i, data in enumerate(self.data):
            with open(path+f'\\{i}.pkl', 'wb') as f:
                pickle.dump(data, f)

    def load(self, path):
        files = os.listdir(path)
        self.data = []
        for file in files:
            with open(path + '\\' + file, 'rb') as f:
                data  = pickle.load(f)
            self.data.append(data)
             

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]         


class GCPL_dataset_resampled2(Dataset):
    def __init__(self, GCPL_dataset, minLength = 0, maxLength= np.inf, sampling = 60, load=False):
        """sampling - number of seconds."""
        if load==False:
            self.data = []
            self.delta = dt.timedelta(seconds=sampling)
            for i in range(len(GCPL_dataset)):
                df, SoH, SoH_n = GCPL_dataset[i]
                if not ((df.shape[0] < maxLength) and (df.shape[0] > minLength)):
                    continue
                if np.isnan(np.sum(SoH)):
                    continue
                if SoH <5:
                    continue
                curr = {'SoH': SoH}
                curr['SoH_next'] = SoH_n
                time = pd.DatetimeIndex(df['time/s'].apply(dt.datetime.fromtimestamp))
                df.set_index(time, inplace=True)
                df = df.resample(self.delta).mean()
                df.interpolate(method='linear', inplace=True)
                curr['E'] = df['Ewe/V'].to_numpy()
                curr['I'] = df['<I>/mA'].to_numpy()
                curr['datetime'] = df.index.to_numpy()
                self.data.append(curr)
        else:
            self.load(GCPL_dataset)
    def del_outliers(self, n):
        """n = max number of dots."""
        new_data = []
        for curr in self.data:
            if len(curr['E'])< n:
                new_data.append(curr)
        self.data = new_data

    def save(self, path, overwrite=False):
        if overwrite and os.path.exists(path):
            shutil.rmtree(path)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path+f'/data.pkl', 'wb') as f:
            pickle.dump(self.data, f)

    def load(self, path):
        self.data = []
        with open(path + '/data.pkl', 'rb') as f:
            self.data  = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]   

class GCPL_dataset_resampled3(Dataset):
    def __init__(self, loadpath):
        self.load(loadpath)

    def save(self, path, overwrite=False):
        if overwrite and os.path.exists(path):
            shutil.rmtree(path)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path+f'/data.pkl', 'wb') as f:
            pickle.dump(self.data, f)

    def load(self, path):
        self.data = []
        with open(path + '/data.pkl', 'rb') as f:
            self.data  = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __delitem__(self, index):
        del self.data[index]    

    def append(self, item):
        self.data.append(item)     

class ProgressPlotter:
  def __init__(self) -> None:
    self._history_dict = defaultdict(list)

  def add_scalar(self, tag: str, value)-> None:
    self._history_dict[tag].append(value)

  def display_keys(self,ax,tags):
    if isinstance(tags, str):
      tags = [tags]
    history_len = 0
    ax.grid()
    for key in tags:
      ax.plot(self._history_dict[key], marker="X",label=key)
      history_len = max(history_len,len(self.history_dict[key]))
      if len(tags) > 1:
        ax.legend(loc="lower left")
      else:
        ax.set_ylabel(key)
    ax.set_xlabel('step')
    # ax.set_xticks(np.arange(history_len))
    # ax.set_xticklabels(np.arange(history_len))
  
  def display(self, groups = None): 
    # groups list ofkeys like [['loss_train','loss_val'],['accuracy']]
    clear_output()
    n_groups = len(groups)
    fig, ax = plt.subplots(n_groups, 1, figsize=(12, 3*n_groups))
    if n_groups == 1:
      ax = [ax]
    for i, keys in enumerate(groups):
      self.display_keys(ax[i],keys) 
    fig.tight_layout()
    plt.show()

  @property
  def history_dict(self):
    return dict(self._history_dict)
  

class ModelHandler:
    def __init__(self, model, loss=1e6, comments = 'No comments', path = './models/v2', kfold= False ):
        # self.train = dataset_train
        # self.val = dataset_val
        self.model = model
        # self.optimizer = optimizer
        self.save_best_model()
        self.best_loss = loss
        self.pp_list = []
        self.comments = comments
        self.path = path
        self.epochs = 0
        self.kfold = kfold
        if not os.path.exists(path):
            os.makedirs(path)
        if len(next(os.walk(path))[1]) > 0 and kfold==False:
            self.path += '/' + str(len(next(os.walk(path))[1]))


    def save_best_model(self):
        self.best_model = copy.deepcopy(self.model)
        # self.best_optimizer = copy.deepcopy(self.optimizer)

    def check_loss(self, loss, check_every = 1):
        """check_loss Check if loss better than current loss

        Args:
            loss (float): loss after training
        """
        self.epochs += 1 *check_every
        if self.best_loss > loss:
            self.save_best_model()
            self.best_loss = loss
    
    def add_pp(self, pp):
        """add_pp Adds Progress plotter to inner logic

        Args:
            pp (ProgressPlotter): Result of 
        """
        self.pp_list.append(pp)
    
    def display(self):
        """display works only for 'loss_train' and 'loss_val'.
        """
        pp = ProgressPlotter()
        pp._history_dict['loss_train'] = self.pp_list[0].history_dict['loss_train']
        pp._history_dict['loss_val'] = self.pp_list[0].history_dict['loss_val']
        for p in self.pp_list[1:]:
            pp._history_dict['loss_train'] += p.history_dict['loss_train']
            pp._history_dict['loss_val'] += p.history_dict['loss_val']
        pp.display([['loss_train','loss_val']])
    
    def save(self, path=None, comment = None, kfold_number = None ):
        if path:
            self.path=path
        if comment:
            self.comments += comment
        checkpoint = {'model_state_dict': self.best_model.state_dict(),
            # 'optimizer_state_dict': self.best_optimizer.state_dict(),
            'loss': self.best_loss,
            'comment': self.comments, 
            'pp_list': self.pp_list,
            'epochs': self.epochs,
            # 'train_set': self.train,
            # 'val_set': self.val
            }
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        num_checkpoints = len(next(os.walk(self.path))[2])
        if self.kfold:
            if not os.path.exists(self.path+f'/{kfold_number}'):
                os.makedirs(self.path+f'/{kfold_number}')
            num_checkpoints = len(next(os.walk(self.path+f'/{kfold_number}'))[2])
            torch.save(checkpoint, self.path+f'/{kfold_number}'+ '/checkpoint_'+str(num_checkpoints)+'.pt')
        else:
            torch.save(checkpoint, self.path+ '/checkpoint_'+str(num_checkpoints)+'.pt')
    
    def load(self, path):
        if path.split('.')[-1] == 'pt':
            self.path = '/'.join(path.split('/')[:-1])
            self.name = path.split('/')[-1]
        else:
            self.path = path
            files = next(os.walk(self.path))[2]
            x = lambda x: int(x.split('_')[-1].split('.')[0])
            files.sort(key=x)
            self.name = files[-1]
        checkpoint = torch.load(self.path+ '/' + self.name, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict']) 
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
        self.best_model = copy.deepcopy(self.model) 
        self.best_loss = checkpoint['loss']
        self.comments = checkpoint['comment']
        if 'pp_list' in checkpoint.keys():
            self.pp_list = checkpoint['pp_list']
        else:
            self.pp_list = []
        if 'epochs' in checkpoint.keys():
            self.epochs = checkpoint['epochs']
        else:
            self.epochs = 0
        if 'train_set' in checkpoint.keys():
            self.train = checkpoint['train_set']
        else:
            self.train = None
        if 'val_set' in checkpoint.keys():
            self.val = checkpoint['val_set']
        else:
            self.val = None
        

def collate_batch(batch):
    sample_list = []
    label_list = []
    for i in batch:
        sample = np.stack([i['E'], i['I']],axis=-1)
        sample_list.append(torch.tensor(sample, dtype=torch.float32))
        label_list.append(i['SoH'])
    sequence_pad = nn.utils.rnn.pad_sequence(sample_list)
    labels = torch.tensor(label_list, dtype=torch.float32)
    return sequence_pad, labels    

class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1, bidir=False):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size, hidden_size=hidden_size , num_layers=num_layers,bidirectional =bidir)
        self.fc = nn.Linear(hidden_size*num_layers*(1 + bidir), 1)  

    def forward(self, x):
        # print(x.dtype)
        _, h = self.rnn(x)
        h = torch.permute(h, (1,0,2))
        y = self.fc(h.flatten(1,-1))
        return y.flatten()
    

# basic random seed



DEFAULT_RANDOM_SEED = 2023

def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
# tensorflow random seed 

def seedTF(seed=DEFAULT_RANDOM_SEED):
    tf.random.set_seed(seed)
    
# torch random seed

def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
      
# basic + tensorflow + torch 
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTF(seed)
    seedTorch(seed)

def prob(value, counts,borders):
    for i in range(len(counts)):
        if borders[i] <= value <= borders[i+1]:
            return 1/counts[i]

def statistics(dataset):
    info = pd.DataFrame({
                'Pouch':[],
                'SoH':[],
                "Len":[],
                })
    for i, cycle in enumerate(dataset):
        one = len(cycle['E'])
        soh = cycle['SoH']*50+50
        if np.any(np.isnan(cycle['E'])):
            print(i)
        pouch = cycle['Pouch']
        info.loc[len(info)] = [pouch, soh, one]
    stats = info['Pouch'].value_counts()
    pouches = stats.index.tolist()
    soh = {}
    for pouch in pouches:
        min_soh = info[info["Pouch"] == pouch]['SoH'].min()
        soh[pouch] = min_soh
    soh = pd.DataFrame.from_dict(soh, orient='index', columns=['SoH'])
    soh['Cycles'] =stats.values
    soh.sort_index(inplace=True)
    return soh, info

def balancing(dataset, num_of_splits=5):
    #Creating splits for dataset balancing
    soh, info = statistics(dataset)
    counts, borders, _ = plt.hist(info.SoH, num_of_splits)
    probs = []
    for i, cycle in enumerate(dataset):
        proba = prob(cycle['SoH']*50+50, counts, borders)
        probs.append(proba)
    probs = np.array(probs)
    return probs
