import os, glob
import pandas as pd

class Paths(object):
    def __init__(self, root_path=None, audios_path=None, bird_name=None):
        if root_path==None: self.root = ".\\examples\\"
        else:               self.root = root_path
        #self.base     = "{}birdsongs\\".format(self.root) 
        self.auxdata  = '{}auxiliar_data\\'.format(self.root)
        self.results  = '{}results\\'.format(self.root) 
        self.examples = '{}audios\\'.format(self.results)  # write audios folder
        
        if audios_path==None:
            self.audios      = '{}audios\\'.format(self.root)     # wav folder
            self.sound_files = glob.glob(os.path.join(self.audios, '*wav'))
        else:
            if "ebird" in audios_path: 
                search_by, name_col = "Scientific Name", "ML Catalog Number"
            elif "humbolt":                
                search_by, name_col = "Nombre científico", "Número de Catálogo"
            
            self.audios = audios_path
            data = pd.read_csv(self.audios+'spreadsheet.csv')
            data.dropna(axis = 0, how = 'all', inplace = True)
            self.data = data.convert_dtypes() #.dtypes  

            if bird_name!=None:
                    data_bird = self.data[search_by].str.contains(bird_name)
                    self.indexes   = data_bird[data_bird==True].index
            else:   self.indexes   = self.data.index
            files_names = self.data[name_col][self.indexes].astype(str).str.cat([".wav"]*self.indexes.size)
            audios_paths = pd.Series([audios_path]*files_names.size)
            if files_names.values.size!=0: 
                self.sound_files = audios_paths.str.cat(files_names.values).values
            else: self.sound_files = []
        
        self.no_files    = len(self.sound_files)
        
        if not os.path.exists(self.results):    os.makedirs(self.results)
        if not os.path.exists(self.examples):   os.makedirs(self.examples)
        print("The folder has {} songs".format(self.no_files))
        
    def ShowFiles(self):
        [print(str(i)+"-"+self.sound_files[i]) for i in range(len(self.sound_files))];