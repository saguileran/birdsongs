---
permalink: /use/
layout: single
author_profile: false
title: "Use"
excerpt: "Installation and use of birdsongs."
last_modified_at: 2023-01-05T11:59:26-04:00
toc: true
---

To use 

## Requirments

birdsong is implemented in python 3.8. It requires:

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
- lazy_loader


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
python -m pip install -r ./requirements.txt
```

You can now use the package in a python terminal opened at the birdsongs folder. 


<!--
To use the package from any folder install the repository, this can be done with any of the two following lines

```bat
python .\setup.py install
pip install .
```
-->
That's all! 

## Tutorial NoteBooks

Take a look at the tutorials notebooks for basic uses:

| Name                                        | Description                                           |
| ------------------------------------------- | ----------------------------------------------------- |
| [BirdSongs.ipynb](https://github.com/saguileran/birdsongs/blob/main/BirdSongs.ipynb) | General use, generate a whole birdsong (several syllables). |
| [Syllable.ipynb](https://github.com/saguileran/birdsongs/blob/main/syllable.ipynb) | Syllable generation, define and generate a syllable from a recorded birdsong. |
| [Motor Gestures.ipynb](https://github.com/saguileran/birdsongs/blob/main/motor-gestures.ipynb) | Physical model. |


## Use

### Define

Import the package as `bs`

```python
import birdsongs as bs
from birdsongs.utils import *
```  
  
Define a ploter and paths objects, optionally you can specify the audio folder or enable to save figures 

```python
# audios = "path\to\audios" # default examples/audios/
# root = "path\to\audios"
# bird_name = "path\to\audios"

ploter = bs.Ploter()  # save = True to save the figures
paths  = bs.Paths()   # root, audios_path, bird_name
```

display the audios found with `paths.ShowFiles()`, or if the folder has an spreadsheet `paths.data`.

**BirdSong**
  
Define and plot the audio of a birdsong 

```python
birdsong = bs.BirdSong(paths, no_file=2, NN=1024) # tlim=(t0,tend) you can also give a time interval or frequency limits flim=(f0,fmax)
ploter.Plot(birdsong)
birdsong.Play()    # in notebook useAudioPlay(birdsong)
```

**Syllables**
  
Define the syllables using time intervals of interest from the whole bird song. You can choose the points by with the function `ploter.Plot` changing the value of `SelectTime_on=True`
    
```python
ploter.Plot(birdsong, FF_on=False, SelectTime_on=True)
time_intervals = Positions(ploter.klicker)
time_intervals

syllable = bs.Syllable(birdsong, tlim=time_intervals[0], umbral_FF=birdsong.umbral_FF, NN=birdsong.NN, Nt=30, ide="syllable")
syllable.no_syllable = 10
ploter.Plot(syllable, FF_on=True);
``` 
  
### Solve
  
The final step is to define the optimizer object to generate the synthetic syllable (song), solve the optimization problem. For example, to generate the synthetic syllable (or birdsong) with the time intervals defined previously 

```python
brute          = {'method':'brute', 'Ns':11}              # method of optimization, Ns is the number of grid points
optimizer     = bs.Optimizer(syllable, method_kwargs=brute)
optimal_gamma = optimizer.OptimalGamma(syllable)

optimizer.OptimalParams(syllable, Ns=11)

# optimizer      = bs.Optimizer(birdsong, method_kwargs=brute)   # optimizer object, they will save all the optimal parameters
#syllable, synth_syllable = optimizer.SongByTimes(time_intervals)   # find the best syllables for each syllable
```
    
define the optimal syllable (synthetic)

```python
synth_syllable = syllable.Solve(syllable.p)
```


### Visualize
  
Visualize and write the synthetic optimal audio 
    
```python
# ploter.Plot(synth_syllable);
# ploter.PlotVs(synth_syllable);
# ploter.PlotAlphaBeta(synth_syllable);
# ploter.Syllables(syllable, synth_syllable);
ploter.Result(syllable, synth_syllable);


birdsong.WriteAudio();  synth_syllable.WriteAudio()
```
  
### Note  
  
To find a single synthetic syllable (or chunck) the process is the following. Nevertheless, to define a syllable (or chunck) you must have defined a bird song (syllable) object. 

```python
syllable  = bs.Syllable(birdsong)               # define the syllable. you can also give a time or frequency interval: flim=(fmin,fmax), tlim=(t0,tend)

brute     = {'method':'brute', 'Ns':11}     # define optimization method and its parameters
optimizer = bs.Optimizer(syllable, brute)   # define optimizer to the syllable object

optimizer.optimal_gamma                     # find teh optimal gamma over the whole bird syllables
obj = syllable  # birdsong
optimizer.OptimalParams(obj, Ns=11)         # find optimal alpha and beta parameters
    
Display(obj.p)                              # display optimal problem parameters
obj_synth_optimal = obj.Solve(obj.p)        # generate the synthetic syllable with the optimal parameters set
    
ploter.Syllables(obj, obj_synth_optimal)    # plot real and synthetic songs, sound waves and spectrograms
ploter.PlotAlphaBeta(obj_synth_optimal)     # plot alpha and beta parameters in function of time (just syllable has this attributes)
ploter.Result(obj, obj_synth_optimal)       # plot the spectrograms, scores and features of both objects, the real and synthetic
    
bird.WriteAudio();  synth_bird.WriteAudio() # write both objects, real and synthetic
```
    
The repository has some audio examples, in [examples/audios](https://github.com/saguileran/birdsongs/tree/main/examples/audios) folder. You can download and store your own audios in the same folder or enter the audio folder path to the Paths object. The audios **must** be in WAV format or birdosngs will not import them, we suggest use [audacity](https://www.audacityteam.org/) to convert the audios without any problem.

    
<!---
and then add to python 

```bat
pip install -e birdsongs
python -m pip install -e birdsongs
```
-->




[Use the package]({{ "https://github.com/saguileran/birdsongs" | relative_url }}){: .btn .btn--success .btn--large}



---

asdasd