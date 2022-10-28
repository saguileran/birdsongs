from song_class import * 
# ------------------- PATHS ------------------------
root_path     = "C:\\Users\\sebas\\Documents\\GitHub\\" # root_path = '/home/siete/Downloads/audios/'
base_path     = "{}BirdSongs\\".format(root_path) 
audios_path   = '{}Audios\\'.format(root_path)
auxdata_path  = '{}AuxiliarData\\'.format(base_path)
results_path  = '{}Results\\'.format(base_path) 
examples_path = '{}Results\\Examples\\'.format(base_path)
sound_files   = glob.glob(os.path.join(audios_path, '*wav')) #'*'+birdname+'*wav'   

if not os.path.exists(results_path):    os.makedirs(results_path)
if not os.path.exists(examples_path):    os.makedirs(examples_path)