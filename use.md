---
permalink: /use/
layout: splash
search: true
author_profile: false
title: "Use"
excerpt: "Installation and use of birdsongs."
last_modified_at: 2024-01-13T11:59:26-04:00
toc: true
---

Here you will find a tutorial on how to download, install, and use the `birdsongs` package.

# Installation

## Requirments

`birdsong` is implemented in Python 3.8 but it works for any latest Python version. The package requirements can be found at [requirements.txt](https://github.com/saguileran/birdsongs/blob/main/requirements.txt).

## Setup Python Environment

If you are using Anaconda, it is always a good practice to create a new environment to test new packages. For that, use the following command lines:

```bash
conda create -n birdsongs python=3.12
conda activate birdsong
```
When you finished using the birdsongs package just enter `conda deactivate` to disable the birdsongs environment. 

This can be also done in vsc using Python environments.
    
## Downloading

To use birdsongs, clone the main branch of the repository and go to its root folder.

```bat
git clone  https://github.com/saguileran/birdsongs.git --single-branch
cd birdsongs
```

You can clone the whole repository using `git clone https://github.com/saguileran/birdsongs.git` but since it is huge only the main branch is enough. To change the branch use the command `git checkout branch_name` following the branch name of interest.

The next step is to install the required packages, any of the following command lines will work:

```bat
python -m pip install -r ./requirements.txt
```

If you are using a version of Python higher than 3.10, to listening the audios you must execute 

```python
pip install playsound@git+https://github.com/taconi/playsound
```

In other case just execute `pip install playsound`.

<!--
You can now use the package in a Python terminal opened in the birdsongs folder. 

To use the package from any folder install the repository, this can be done with any of the two following lines
-->

Now, install the birdsong package using any one of the following lines

```bat
python setup.py install
python -m pip install .
```

or using pip (not recomended)
```bat
pip install -e .
pip install .
```

That's all. Now let's create a synthetic birdsong!

Take a look at the tutorials notebooks for basic uses: physical model implementation, [motor-gestures.ipynb](https://github.com/saguileran/birdsongs/blob/main/tutorials/motor-gestures.ipynb); define and generate a syllable from a recorded birdsong, [syllable.ipynb](https://github.com/saguileran/birdsongs/blob/main/tutorials/syllable.ipynb); or to generate a whole birdsong, several syllables, [birdsong.ipynb](https://github.com/saguileran/birdsongs/blob/main/tutorials/birdsong.ipynb),

# Use

## Define Objects

Import the package as `bs` and its utilities

```python
import birdsongs as bs
from birdsongs.util import *
```

### Path and Plotter
  
First, define the plotter and path objects, optionally you can specify the audio folder or enable the plotter to save figures. Remember you are at the birdsongs folder (../birdsongs)

```python
root    = "examples" # "path\\to\\repository\\' 
audios  = 'audios'    # "path\\to\\audios\\'
results = "results"   # "path\\to\\results\\'

paths  = bs.Paths(root, audios, results, catalog=False)      # root_path, audios_path, catalog
ploter = bs.Ploter(save=True)   # to save figures save=True 
```

Displays the audio file names found with the paths.AudiosFiles(True) function, if the folder has a spreadsheet.csv file this function displays all the information about the files inside the folder otherwise it displays the audio file names found.

### BirdSong
  
Define and plot the wave sound and spectrogram of a birdsong object, for example the audio file "XC11293"

```python
birdsong = bs.BirdSong(paths, file_id="574179401", NN=1024, umbral_FF=1., Nt=500,
                       #tlim=(t0,tend), flim=(f0,fmax) # other features
                      )
ploter.Plot(birdsong, FF_on=False)  # plot the wave sound and spectrogram without FF
birdsong.Play()                     # listen to the birdsong
```

### Syllables
  
Define the syllables using time intervals of interest from the whole birdsong. You can choose the points with the `ploter.Plot()` function by changing the value of `SelectTime_on` to `True`
    
```python
ploter.Plot(birdsong, FF_on=False, SelectTime_on=True) # selct 
time_intervals = Positions(ploter.klicker)             # save the points selected in an array
time_intervals                                         # displays

syllable = bs.Syllable(birdsong, tlim=time_intervals[0], Nt=5, NN=256, umbral_FF=1.05,
                       no_syllable=0, type="intro")
ploter.Plot(syllable, FF_on=True);
syllable.Play()
``` 
  
## Solve
  
Now let's define the optimizer object to generate the synthetic syllable, i.e., to solve the optimization problem. For example, to generate the synthetic syllable (or chunk) from the previously selected time interval.

```python
brute_kwargs = {'method':'brute', 'Ns':11}          # optimization method,  Ns is the number of grid points
optimizer    = bs.Optimizer(syllable, brute_kwargs) # optimizer object
optimal_gm   = optimizer.OptimalGamma(syllable)     # find optimal gamma (time scale constant) 

optimizer.OptimalParams(syllable, Ns=11)            # find optimal parameters coefficients
#syllable, synth_syllable = optimizer.SongByTimes(time_intervals)   # find optimal parameters over several time intervals
```
    
Then, define the synthetic syllable object with the optimal values found above.


```python
synth_syllable = syllable.Solve(syllable.p)
```

## Visualize
  
Finally, visualize and write the optimal synthetic audio.
    
```python
ploter.Plot(synth_syllable);                # sound wave and spectrogram of the synthetic syllable
ploter.PlotVs(synth_syllable);              # physical model variables over the time
ploter.PlotAlphaBeta(synth_syllable);       # motor gesture curve in the parametric space
ploter.Syllables(syllable, synth_syllable); # synthetic and real syllables
ploter.Result(syllable, synth_syllable);    # scoring variables and other spectral features

birdsong.WriteAudio();  synth_syllable.WriteAudio(); # write both audios at ./examples/results/Audios
```
  
## Note  
  
To generate a single synthetic syllable (or chunk) you must have defined a birdsong (or syllable) and the process is as follows:

1. Define a path object.
2. Define a birdsong object using the above path object, it requires the audio file id. You can also enter the length of the window FFT and the umbral (threshold) for computing the FF, between others.
3. Select or define the time intervals of interest.
4. Define an optimization object with a dictionary of the method name and its parameters.
5. Find the optimal gammas for all the time intervals, or a single, and average them.
6. Find and export the optimal labia parameters for each syllable, the motor gesture curve.
7. Generate synthetic birdsong from the previous control parameters found.
8. Visualize and save all the syrinx, scoring, and result variables.
9. Save both synthetic and real syllable audios.

The repository has some audio examples, in the ./examples/audios folder. You can download and store your audios in the same folder or enter the audio folder path to the Paths object.

The audios can be in WAV or MP3 format. If you prefer WAV format, we suggest using Audacity to convert the audios without any issue.
<!--

```python
syllable  = bs.Syllable(birdsong)           # additional options: flim=(fmin,fmax), tlim=(t0,tend) 

brute     = {'method':'brute', 'Ns':11}     # define optimization method and its parameters
optimizer = bs.Optimizer(syllable, brute)   # define optimizer to the syllable object

optimizer.optimal_gamma                     # find the optimal gamma over the whole bird syllables
obj = syllable                              # birdsong or chunk
optimizer.OptimalParams(obj, Ns=11)         # find optimal alpha and beta parameters
    
Display(obj.p)                              # display optimal problem parameters
obj_synth_optimal = obj.Solve(obj.p)        # generate the synthetic syllable with the optimal parameters set
    
ploter.Syllables(obj, obj_synth_optimal)    # plot real and synthetic songs, sound waves, and spectrograms
ploter.PlotAlphaBeta(obj_synth_optimal)     # plot alpha and beta parameters in function of time (just syllable has these attributes)
ploter.Result(obj, obj_synth_optimal)       # plot the spectrograms, scores, and features of both objects, the real and synthetic
    
bird.WriteAudio();  synth_bird.WriteAudio() # write both objects, real and synthetic
```
-->
    
The repository has some audio examples in the folder [./examples/audios](https://github.com/saguileran/birdsongs/tree/main/examples/audios). You can download and store your audios in the same folder or enter another audio folder path to the Paths object, the package also has a function to download audios from Xeno-Canto: birdsong.util.DownloadXenoCanto().

>[!IMPORTANT]
>The audios **must** be in WAV format or birdsongs will not import them, we suggest using [Audacity](https://www.audacityteam.org/) to convert the audios without any problem.

---

Now you can generate a synthetic syllable using a recorded birdsong.
