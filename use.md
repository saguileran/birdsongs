---
permalink: /use/
layout: single
search: true
author_profile: false
#title: "Use"
excerpt: "Installation and use of birdsongs."
last_modified_at: 2023-01-05T11:59:26-04:00
toc: true
---

  
# Installation

## Requirments

`birdsong` is implemented in python 3.8 and requires:


- librosa
- lmfit
- scipy
- sympy
- numpy
- pandas
- matplotlib
- playsound
- PeakUtils
- mpl_pan_zoom
- mpl_point_clicker
- scikit_learn
- scikit_maad
- setuptools
- ipython
- pygobject
- ffmpeg

    
## Downloading

In order to use birdsongs, clone the repository and enter to the folder repository

```bat
git clone https://github.com/saguileran/birdsongs.git
cd birdsongs
```
you can verify the current branch with the command `git branch -a`. You have to be in `main` branch, to change the branch use the command `git checkout main`.

The next step is to install the required packages, any of the following commands lines will work

```bat
pip install -r ./requirements.txt
# python -m pip install -r ./requirements.txt  # (equivalent)
```
<!--
You can now use the package in a python terminal opened at the birdsongs folder. 

To use the package from any folder install the repository, this can be done with any of the two following lines
-->

Install the birdsong package

```bat
python .\setup.py install
```

That's all! 
   
<!---
and then add to python 

```bat
pip install -e birdsongs
python -m pip install -e birdsongs
```
-->

Take a look at the tutorials notebooks for basic uses: physical model implementation, [motor-gestures.ipynb](./tutorials/motor-gestures.ipynb); define and generate a syllable from a recorded birdsong, [syllable.ipynb](./tutorials/syllable.ipynb); or to generate a whole birdsong, several syllables, [birdsong.ipynb](./tutorials/birdsong.ipynb),

# Use

## Define

Import the package as `bs` 

```python
import birdsongs as bs
```  
  
Define the ploter and paths objects, optionally you can specify the audio folder or enable to save figures 

```python
# audios = "path\to\audios"     # default examples/audios/
# root = "path\to\audios"       # default ./
# bird_name = "path\to\audios"  # default None

ploter = bs.Ploter(save=True)  # images are save at ./examples/results/Images/
paths  = bs.Paths()            # root, audios_path, bird_name
```

Displays the audios found with the `paths.AudiosFiles()` function, if the folder has a *spreadsheet.csv* file this functions displays all the information about the files inside the folder.

**BirdSong**
  
Defining and plotting the wave sound and spectrogram of a birdsong object

```python
birdsong = bs.BirdSong(paths, no_file=0, NN=1024, umbral_FF=1.0,
                       #tlim=(t0,tend), flim=(f0,fmax) # other features
                      )
ploter.Plot(birdsong, FF_on=False)  # plot the wave sound and spectrogram
birdsong.Play()                     # in notebook useAudioPlay(birdsong)
```

**Syllables**
  
Define the syllables using time intervals of interest from the whole birdsong. You can choose the points with the `ploter.Plot()` function by changing the value of `SelectTime_on` to `True`
    
```python
ploter.Plot(birdsong, FF_on=False, SelectTime_on=True) # selct 
time_intervals = Positions(ploter.klicker)             # save 
time_intervals                                         # displays

syllable = bs.Syllable(birdsong, tlim=time_intervals[0], NN=birdsong.NN, Nt=30,
                       umbral_FF=birdsong.umbral_FF, ide="syllable")
ploter.Plot(syllable, FF_on=True);
``` 
  
## Solve
  
The last step consists in defining the optimizer object to generate the synthetic syllable (song), solving the optimization problem. For example, to generate the synthetic syllable (or birdsong) with the previously defined time intervals 

```python
brute_kwargs = {'method':'brute', 'Ns':11}          # optimization mehotd. Ns is the number of grid points
optimizer    = bs.Optimizer(syllable, brute_kwargs) # optimizer object
optimal_gm   = optimizer.OptimalGamma(syllable)     # find optimal gamma 

optimizer.OptimalParams(syllable, Ns=11)            # find optimal parameters coefficients
#syllable, synth_syllable = optimizer.SongByTimes(time_intervals)   # find optimal parameters over the time intervals
```
    
define the optimal synthetic syllable object with the values found above

```python
synth_syllable = syllable.Solve(syllable.p)
```

## Visualize
  
Visualize and write the optimal synthetic audio 
    
```python
ploter.Plot(synth_syllable);                # sound wave and spectrogram of the synthetic syllable
ploter.PlotVs(synth_syllable);              # physical model variables over the time
ploter.PlotAlphaBeta(synth_syllable);       # motor gesture curve in the parametric space
ploter.Syllables(syllable, synth_syllable); # synthetic and real syllables
ploter.Result(syllable, synth_syllable);    # scoring variables and other spectral features

birdsong.WriteAudio();  synth_syllable.WriteAudio(); # write both audios at ./examples/results/Audios
```
  
## Note  
  
To generate a single synthetic syllable (chunck) you must have defined a birdsong (syllable), the process is as follows:

1. Define a paths object.
2. Use the previous path obeject to define a birdsong (syllable) object, it also requeries the file number (birdsong for a syllable). Here you can define the window FFT length and the umbral threshold to compute the pitch 
3. Define an optimization object with a dictionary of the method name and its parameters.
4. Find the optimal gamma, for a single syllable or for a set of syllables defined from time intervals.
5. Find the optimal labia parameters, the motor gesture curve.
6. Generate the synthetic birdsong from the previous control parameters found.
7. Plot and save all the syrinx, scoring, and result variables.
8. Write the syllable audios defined both synthetic and real.
<!--
```python
syllable  = bs.Syllable(birdsong)           # additional options: flim=(fmin,fmax), tlim=(t0,tend) 

brute     = {'method':'brute', 'Ns':11}     # define optimization method and its parameters
optimizer = bs.Optimizer(syllable, brute)   # define optimizer to the syllable object

optimizer.optimal_gamma                     # find teh optimal gamma over the whole bird syllables
obj = syllable                              # birdsong or chunck
optimizer.OptimalParams(obj, Ns=11)         # find optimal alpha and beta parameters
    
Display(obj.p)                              # display optimal problem parameters
obj_synth_optimal = obj.Solve(obj.p)        # generate the synthetic syllable with the optimal parameters set
    
ploter.Syllables(obj, obj_synth_optimal)    # plot real and synthetic songs, sound waves and spectrograms
ploter.PlotAlphaBeta(obj_synth_optimal)     # plot alpha and beta parameters in function of time (just syllable has this attributes)
ploter.Result(obj, obj_synth_optimal)       # plot the spectrograms, scores and features of both objects, the real and synthetic
    
bird.WriteAudio();  synth_bird.WriteAudio() # write both objects, real and synthetic
```
-->
    
The repository has some audio examples, in [./examples/audios](https://github.com/saguileran/birdsongs/tree/main/examples/audios) folder. You can download and store your own audios in the same folder or enter the audio folder path to the Paths object, the package also has a function to download audios from Xeno-Canto: birdsong.util.DownloadXenoCanto().

The audios **must** be in WAV format or birdosngs will not import them, we suggest use [Audacity](https://www.audacityteam.org/) to convert the audios without any problem.

    
<!---
and then add to python 

```bat
pip install -e birdsongs
python -m pip install -e birdsongs
```
-->



---

Now you are able to generate a synthetic syllable using a recorded birdsong.