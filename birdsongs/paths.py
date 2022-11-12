import os, glob

class Paths(object):
    def __init__(self, root_path, bird_name=""):
        self.root     = root_path
        self.base     = "{}birdsongs\\".format(self.root) 
        self.auxdata  = '{}auxiliar_data\\'.format(self.base)
        self.results  = '{}results\\'.format(self.base) 
        self.audios   = '{}audios\\'.format(self.root)     # wav folder
        self.examples = '{}audios\\'.format(self.results)  # write audios folder

        if not os.path.exists(self.results):    os.makedirs(self.results)
        if not os.path.exists(self.examples):    os.makedirs(self.examples)

        self.sound_files   = glob.glob(os.path.join(self.audios, '*wav')) #'*'+birdname+'*wav' 
        self.no_files      = len(self.sound_files)
        
        print("The folder has {0} songs".format(self.no_files))