import os, string
import pandas as pd
from pathlib import Path
from maad import sound
import requests
from librosa import load
import numpy as np
from os.path import relpath

#%%
class Paths(object):
    #%%
    def __init__(self, root_path="../examples/", audios_path='audios', results="results", catalog=False): #  bird_name=None,
        
        self.root = Path(root_path)
        self.catalog = catalog
        self.audios = self.root / audios_path
        
        #self.auxdata  = self.root / 'auxiliar_data' #auxiliar_data
        self.results  = self.root / results
        self.MG_param = self.results / "MG"
        self.images   = self.results / "Images"
        self.examples = self.results / 'Audios'  # write audios folder
        
        # create folder in case they do not exist
        if not os.path.isdir(self.results): os.makedirs(self.results)
        if not os.path.isdir(self.MG_param): os.makedirs(self.MG_param)
        if not os.path.isdir(self.images): os.makedirs(self.images)
        if not os.path.isdir(self.examples): os.makedirs(self.examples)
        
        # Find all files in the folder sel.audios
        sound_files_wav = list(self.audios.glob("*.wav"))
        sound_files_mp3 = list(self.audios.glob("*.mp3"))
        sound_files_wav_str = [str(s) for s in sound_files_wav]
        sound_files_mp3_str = [str(s) for s in sound_files_mp3]
        
        self.sound_files = sound_files_wav + sound_files_mp3
        self.files_names = [str(f)[len(str(self.audios))+1:] for f in self.sound_files]
    
        # Read data from a spreadsheet.csv which contains all data about the records 
        if catalog is True:
            data = pd.read_csv(self.audios/'spreadsheet.csv', encoding_errors='ignore')
            data.dropna(axis = 0, how = 'all', inplace = True)
            self.data = data.convert_dtypes()
            self.indexes   = self.data.index
        
            name_col = self.data.columns[0] # first column od spreadsheet.csv
            
            files_names = self.data[name_col][self.indexes].astype(str)
            files_names_f = []
            for i in range(len(files_names)):
                if any(files_names[i] in ext for ext in sound_files_wav_str):
                    files_names_f.append(files_names[i]+".wav")
                elif any(files_names[i] in ext for ext in sound_files_mp3_str):
                    files_names_f.append(files_names[i]+".mp3")
            files_names = files_names_f

            audios_paths = pd.Series([audios_path]*len(files_names))
            
            if len(files_names)!=0: 
                self.sound_files = audios_paths.str.cat(files_names).values
                self.files_names = [f[len(audios_paths[0]):] for f in self.sound_files]
            else: 
                self.sound_files = []; self.files_names = []; 
        self.no_files  = len(self.sound_files)

    #%%
    def ImportParameters1(self, XC=None, no_file=None, no_syllable=None, name=None):

        self.data_param = self.MG_Files()

        if name is not None and XC is None and no_file is None :
            df = self.data_param["name"].str.contains(name,case=False)
            
        if XC is not None and no_file is None and name is None:
            df = self.data_param[self.data_param['id_XC'] == XC]
        if no_file is not None and XC is None and name is None:
            df = self.data_param.iloc[no_file]
        if no_file is None and XC is None:
            df = self.data_param
        if no_file is None and XC is None and name is None:
            df = self.data_param

        if no_syllable is not None: 
            df = df[df['no_syllable'] == str(no_syllable)]
            coef = pd.read_csv(df["coef_path"].values[0]).rename(columns={"Unnamed: 0":"parameter"})
            tlim = pd.Series({"t_ini":coef.iloc[-2].value, "t_end":coef.iloc[-1].value})
            df = pd.concat([df, tlim]).reset_index()
            
            return df#, coef#, motor_gesture
        else:                               # if syllables is None
            coefs, type, out, tlim, NN, umbral_FF, country, state = [], [], [], [], [], [], [], []
            for i in df.index:
                coef = pd.read_csv(self.data_param.iloc[i]["coef_path"], index_col="Unnamed: 0", engine='python')#, encoding = "utf-8") #cp1252
                tlim.append([float(coef.iloc[7].value), float(coef.iloc[8].value)])
                NN.append(int(coef.iloc[9].value))
                umbral_FF.append(float(coef.iloc[10].value))
                type.append(coef.iloc[11].value)
                country.append(coef.iloc[12].value)
                state.append(coef.iloc[13].value)
                coefs.append(coef.iloc[:7].astype('float64'))
            tlim = np.array(tlim)
            
            df = pd.DataFrame({'id_XC':df['id_XC'], 'no_syllable':df['no_syllable'],
            'id':df['id'], 'name':df['name'], 'coef_path':df['coef_path'], 'param_path':df['param_path'],
            'audio_path':df['audio_path'], 's':df['s'], 'fs':df['fs'], 'file_name':df['file_name'],
            't_ini':tlim[:,0], 't_end':tlim[:,1], 'NN':NN, 'umbral_FF':umbral_FF, 'coef':coefs, 'type':type, 'country':country, 'state':state},
            index=df.index)
            
            out = [df.iloc[i] for i in range(len(df.index))]
            self.df = df.reset_index(drop=True, inplace=False)
            
            print("{} files were found.".format(len(df.index)))
            return out, df
        
    #%%
    def ImportParameters(self, no_syllable=None, country_filter=None):

        df = self.MG_Files()
                                    # if syllables is None
        coefs, type, out, tlim, NN, umbral_FF, country, state = [], [], [], [], [], [], [], []
        for i in df.index:
            coef = pd.read_csv(self.data_param.iloc[i]["coef_path"], index_col="Unnamed: 0", engine='python').T#, encoding = "utf-8") #cp1252
            tlim.append([float(coef["t_ini"].value), float(coef["t_end"].value)])
            NN.append(int(coef["NN"].value))
            umbral_FF.append(float(coef["umbral_FF"].value))
            type.append(coef["type"].value)
            country.append(coef["country"].value)
            state.append(coef["state"].value)
            coefs.append(coef[["a0","a1","a2","b0","b1","b2","gm"]].values[0]) # coef.iloc[:7].astype('float64')
        tlim = np.array(tlim)
        
        df = pd.DataFrame({'id_XC':df['id_XC'], 'no_syllable':df['no_syllable'],
        'id':df['id'], 'name':df['name'], 'coef_path':df['coef_path'],
        # 'param_path':df['param_path'], 's':df['s'], 'fs':df['fs'],
        'audio_path':df['audio_path'], 'file_name':df['file_name'],
        't_ini':tlim[:,0], 't_end':tlim[:,1], 'NN':NN, 'umbral_FF':umbral_FF, 'coef':coefs, 'type':type, 'country':country, 'state':state},
        index=df.index)
        
        self.df = df.reset_index(drop=True, inplace=False)
        print("{} files were found.".format(len(self.df.index)))
        
        if country_filter is not None: 
            self.df = self.df[self.df["country"]==country_filter]
        if no_syllable is not None: 
            self.df = self.df[self.df["no_syllable"]==str(no_syllable)]
        
        return self.df
    #%% 
    def MG_Files(self): 
        self.MG_coef = list(self.MG_param.glob("*MG.csv"))
        MG_coef_splited  = [relpath(MG, self.MG_param).replace(" ","").split("-") for MG in self.MG_coef]
        
        id_XC = [x[0] for x in MG_coef_splited]
        no_syllables = [x[-2] for x in MG_coef_splited]
        id = [x[-3] for x in MG_coef_splited]
        name = [x[1]+"-"+x[2] for x in MG_coef_splited]
        audios = [list(self.audios.glob(id+"*"))[0]  for id  in id_XC]
        file_name = [relpath(audio, self.audios) for audio in audios]
        # ss, fss = [], []
        
        # for audio in audios:
        #     s, fs = load(audio, sr=None)
        #     ss.append(s); fss.append(fs);
            
        self.data_param = pd.DataFrame({'id_XC':id_XC , 'no_syllable': no_syllables, 'id': id, 'name':name, 
                                        "coef_path":self.MG_coef, "audio_path":audios, "file_name":file_name})
                                                #"coef_path":self.MG_coef, "s":ss, "fs":fss, 

        return self.data_param

    #%%
    def ShowFiles(self, verbose=True):
        if verbose: print("The folder has {} songs:".format(self.no_files))
        
        if self.catalog: return self.data
        else:            return [name for name in self.files_names]
        
    #%%
    def CalculateAltitude(self):
        altitude = []
        for i in range(1):#len(paths.data)):
            long, lat = self.data["Longitude"][i], self.data["Latitude"][i]
            if type(long)==np.float64 and type(lat)==np.float64:
                query = ('https://api.open-elevation.com/api/v1/lookup'
                        f'?locations={lat},{long}')
                r = requests.get(query).json()
                elevation = r["results"][0]["elevation"]
                altitude.append(elevation)
            else:
                altitude.append("NA")
        return altitude