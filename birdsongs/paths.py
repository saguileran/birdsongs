import os
import pandas as pd
from pathlib import Path

class Paths(object):
    def __init__(self, root_path=None, audios_path=None, bird_name=None):
        if root_path==None: self.root = Path("./examples")
        else:               self.root = Path(root_path)
        #self.base     = "{}birdsongs\\".format(self.root) 
        self.auxdata  = self.root / 'auxiliar_data'
        self.results  = self.root / 'results'
        self.examples = self.results / 'audios'  # write audios folder
        
        if audios_path==None:
            self.audios      = self.root / 'audios'     # wav folder
            self.sound_files = list(self.audios.glob("*.wav"))
            self.files_names = [str(f)[len(str(self.audios))+1:] for f in self.sound_files]
        else:
            if "ebird" in audios_path: 
                search_by, name_col = "Scientific Name", "ML Catalog Number"
            elif "humbolt" in audios_path:
                search_by, name_col = "Nombre científico", "Número de Catálogo"
            elif "xeno" in audios_path:
                search_by, name_col = "Scientific Name", "File Name"
            
            self.audios = audios_path
            data = pd.read_csv(self.audios+'spreadsheet.csv', encoding_errors='ignore')
            data.dropna(axis = 0, how = 'all', inplace = True)
            self.data = data.convert_dtypes() #.dtypes  

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
        
        self.no_files    = len(self.sound_files)
            
        self.results.mkdir(parents=True, exist_ok=True)
        self.examples.mkdir(parents=True, exist_ok=True)
        
    def ShowFiles(self):
        print("The folder has {} songs:".format(self.no_files))
        [print(str(i)+"-"+str(self.files_names[i])) for i in range(self.no_files)];