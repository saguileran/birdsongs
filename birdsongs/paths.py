import os
import pandas as pd
from pathlib import Path
from maad import sound
import numpy as np
from os.path import relpath

#%%
class Paths(object):
    #%%
    def __init__(self, root_path=None, audios_path=None, bird_name=None):
        if root_path==None: self.root = Path("../examples/")
        else:               self.root = Path(root_path)
        
        self.auxdata  = self.root / 'auxiliar_data'
        self.results  = self.root / 'results'
        self.MG_param = self.results / "MotorGestures-parameters"
        self.images   = self.results / "Images"
        self.examples = self.results / 'Audios'  # write audios folder
        
        if audios_path==None:
            self.audios      = self.root / 'audios'     # wav folder
            sound_files_wav = list(self.audios.glob("*.wav"))
            sound_files_mp3 = list(self.audios.glob("*.mp3"))
            self.sound_files = sound_files_wav + sound_files_mp3
            self.files_names = [str(f)[len(str(self.audios))+1:] for f in self.sound_files]
            #self.data = pd.Series(self.files_names)
        else:
            if "ebird" in audios_path: 
                search_by, name_col = "Scientific Name", "ML Catalog Number"
            elif "humbolt" in audios_path:
                search_by, name_col = "Nombre científico", "Número de Catálogo"
            elif "xeno" in audios_path:
                search_by, name_col = "Scientific Name", "File Name"
            
            self.audios = audios_path

            if bird_name!=None:
                    data_bird = self.data[search_by].str.contains(bird_name, case=False)
                    self.indexes   = data_bird[data_bird==True].index
            else:   self.indexes   = self.data.index
            
            files_names = self.data[name_col][self.indexes].astype(str).str.cat([".wav"]*self.indexes.size)
            audios_paths = pd.Series([audios_path]*files_names.size)
            
            if files_names.values.size!=0: 
                self.sound_files = audios_paths.str.cat(files_names.values).values
                self.files_names = [f[len(audios_paths[0]):] for f in self.sound_files]
            else: 
                self.sound_files = []; self.files_names = [];             
        
        data = pd.read_csv(self.audios/'spreadsheet.csv', encoding_errors='ignore')
        data.dropna(axis = 0, how = 'all', inplace = True)
        self.data = data.convert_dtypes() #.dtypes  

        self.no_files    = len(self.sound_files)
            
        #self.results.mkdir(parents=True, exist_ok=True)
        #self.examples.mkdir(parents=True, exist_ok=True)
    
    #%%
    def AudioFiles(self):
        print("The folder has {} songs:".format(self.no_files))
        return self.data
        #[print(str(i)+"-"+str(self.files_names[i])) for i in range(self.no_files)];

    #%%
    def ImportParameters(self, XC=None, no_file=None, no_syllable=None, name=None):

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
            df = pd.concat([df, tlim])
            motor_gesture = pd.read_csv(df["param_path"].values[0]).drop(columns="Unnamed: 0")
            
            return df, coef, motor_gesture
        else:                               # if syllables is None
            out, tlim, NN, umbral_FF = list(), list(), list(), list()
            for i in df.index:
                coef = pd.read_csv(self.data_param.iloc[i]["coef_path"]).rename(columns={"Unnamed: 0":"parameter"})
                tlim.append([coef.iloc[-4].value, coef.iloc[-3].value])
                NN.append(int(coef.iloc[-2].value))
                umbral_FF.append(coef.iloc[-1].value)
                #self.data_param.iloc[i] = pd.concat([self.data_param.iloc[i], tlim])

                motor_gesture = pd.read_csv(self.data_param.iloc[i]["param_path"]).drop(columns="Unnamed: 0")
            
            tlim = np.array(tlim)
            df["t_ini"] = tlim[:,0]; df["t_end"] = tlim[:,1];
            df["NN"] = NN;  df["umbral_FF"] = umbral_FF; 
            
            #df = pd.DataFrame(df, index=np.arange(len(df)))
            for i in range(len(df.index)):
                out.append([df.iloc[i]["no_syllable"], df.iloc[i], coef, motor_gesture])
            self.df = df #.reindex(np.arange(len(df.index)))
            #self.df1 = self.df.copy(self.df, index=np.arange(len(df.index)))

            print("{} files were found.".format(len(df.index)))
            return out
        

    #%% 
    def MG_Files(self): 
        self.MG_coef = list(self.MG_param.glob("*MG-coef.csv"))
        self.MG_par = list(self.MG_param.glob("*MG.csv"))
        
        MG_coef_splited  = [relpath(MG, self.MG_param).replace(" ","").split("-") for MG in self.MG_coef]
        MG_param_splited = [relpath(MG, self.MG_param).replace(" ","").split("-") for MG in self.MG_par]

        id_XC = [x[0] for x in MG_param_splited]
        no_syllables = [x[-2] for x in MG_param_splited]
        id = [x[-3] for x in MG_param_splited]
        name = [x[1]+"-"+x[2] for x in MG_param_splited]
        audios = [list(self.audios.glob(id+"*"))[0]  for id  in id_XC]
        ss, fss = [], []
        for audio in audios:
            s, fs = sound.load(audio)
            ss.append(s); fss.append(fs);
        self.data_param = pd.DataFrame({'id_XC':id_XC , 'no_syllable': no_syllables, 'id': id, 'name':name, 
                                        "coef_path":self.MG_coef, "param_path":self.MG_par, "audio_path":audios,
                                        "s":ss, "fs":fss})
        
        return self.data_param
