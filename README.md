<a href="https://github.com/saguileran/birdsongs/"><img src="./assets/img/logo.png" width="500"></a>


# birdsongs

A python package to analyze, visualize and generate synthetic birdsongs.


[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/saguileran/birdsongs/main?labpath=BirdSongs.ipynb)


#  Table of Contents
<!---
- [birdsongs](#birdsongs)
- [Table of Contents](#table-of-contents)
--->
- [birdsongs](#birdsongs)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
  - [Requirments](#requirments)
  - [Downloading](#downloading)
  - [Use](#use)
    - [Define](#define)
    - [Solve](#solve)
    - [Visualize](#visualize)
    - [Note](#note)
- [Overview](#overview)
- [Objective](#objective)
- [Repository Contents](#repository-contents)
  - [Physical Model](#physical-model)
  - [Programming Object Oriented (POO)](#programming-object-oriented-poo)
  - [Implementation](#implementation)
- [Results](#results)
- [References](#references)
  - [Literature](#literature)
  - [Software](#software)
  - [Audios Dataset](#audios-dataset)
---
  
  
# Installation

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

Take a look at the tutorials notebooks for basic uses: physical model implementation, [motor-gestures.ipynb](./motor-gestures.ipynb); define and generate a syllable from a recorded birdsong, [syllable.ipynb](./syllable.ipynb);
or to generate a whole birdsong, several syllables, [birdsong.ipynb](./birdsong.ipynb),

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

# Overview

Study and pakcing of the physical model of the motor **gestures of birdsongss**. This model explains the physics of birdsongs by modeling the organs involved in sound production in birds (syrinx, trachea, glottis, Oro-Oesophageal Cavity (OEC), and beak) with oridary differential equations (EDOs). In this work, a Python package is developed to analyze, visualize and generate synthetic birdsongs using the motor gestures model and recorded samples of birdsongs. To automate the model, the problem is formulated as an minimization problem with two control parameters (air sac pressure of the bird’s bronchi and labial tension) and solved using numerical methods, signal processing tools, and numerical optimization. The package is tested by generating comparable birdsongs, solves the minimization problem using recorded samples of birdsongs and comparing the fundamental frequency (denoted as FF, F0, or also called pitch) and spectral conent index (SCI) of both birdsongs.

# Objective

Design, development, and evaluation of a computational-physical model to generating synthetic bird songs from recorded samples.

# Repository Contents

This repository have the documentation, scripts, and results delelop to achive the proposed objective.

The model used, Motor Gestures [1], have been developed by profesog G. Mindlin at the [Dynamical Systems Laboratory](http://www.lsd.df.uba.ar/) (in spanish LSD) of the university of Buenos Aires, Argentina. 

## Physical Model 

Schematic implementation of the physical model **motor gestures of bydsongs**: syrinx, trachea, glotis, OEC, and beak. 

<p align="center"> <img src="./assets/img/model.png" width="700" title="model"></p>

## Programming Object Oriented (POO)

Taking advantege of POO the repetition of long codes is avoid. Using this programming paradigm, the execution and implementation of the model is fast and easy. Five objects ared created to solve the optimization problem and analyze the synthetic syllables:

- **Syllable**: define a object from audio syllable with its tempo and spectral features
- **Optimizer**: define a object to optimize function from method and syllables
- **BirdSong**: define a object to read and split an audio song in syllables 
- **Plot**: define a object to plot real and synthec syllables or songs
- **Paths**: define a object to organize folders location

In order to understand the diagram methodology, the following icons will be used. 

<p align="center">  <img src="./assets/img/objects.png" width="500" alt="methodology"></p>

Each icon is an object with different tasks.

## Implementation

Using the previous objects defined, the optimization problem is solved by following the next diagram 

<p align="center">  <img src="./assets/img/methodology.png" width="600" alt="methodology"></p>

# Results

The model is tested with different syllables of the birdsong of the rufous collared sparrow . Results are located at [examples/examples](./examples/results), images and audios. 

Simple syllable of a birdsong of the Rufous Collared Sparrow

<p align="center">  <img src="./examples/results/ScoresVariables-syllable-4_short_FINCA153_Zonotrichia_capensis_trimed.wav-0.png" width="1000" alt="methodology"></p>

Simple syllable of a birdsong of the Ocellated Tapaculo - Acropternis

<p align="center">  <img src="./examples/results/ScoresVariables-syllable-XC104508 - Ocellated Tapaculo - Acropternis orthonyx.wav-0.png" width="1000" alt="methodology"></p>

<!--
<center>
  Zonotrichia capensis - XC11293 <br>
  <audio src="\examples\results\audios\XC11293 - Rufous-collared Sparrow - Zonotrichia capensis.wav-syllable-0-synth.wav" controls preload></audio>
  <audio src="\examples\results\audios\XC11293 - Rufous-collared Sparrow - Zonotrichia capensis.wav-syllable-0.wav" controls preload></audio>
  
  Euphonia Laniirostris Crassirostris - C541457 <br>
  <audio src="\examples\results\audios\C541457 - Thick-billed Euphonia - Euphonia laniirostris crassirostris.wav-syllable-synth-0.wav" controls preload></audio>
  <audio src="\examples\results\audios\C541457 - Thick-billed Euphonia - Euphonia laniirostris crassirostris.wav-syllable-0.wav" controls preload></audio>
  
</center>
-->

The PDF of the thesis is located in the `dissertation` brach of this repository, <a href="https://github.com/saguileran/birdsongs/blob/dissertation/main.pdf">Design, development, and evaluation of a computational physical model to generate synthetic birdsongs from recorded samples</a>, 

<br><br>


<p align="center">
  <img src="https://github.com/saguileran/birdsongs/blob/gh-pages/assets/img/cover.jpg" width="300" height="400">
  <img src="https://github.com/saguileran/birdsongs/blob/gh-pages/assets/img/under-cover.png" width="300" height="400">
</p>


# References

## Literature

<div class="csl-entry">[1] Amador, A., Perl, Y. S., Mindlin, G. B., &#38; Margoliash, D. (2013). Elemental gesture dynamics are encoded by song premotor cortical neurons. <i>Nature 2013 495:7439</i>, <i>495</i>(7439), 59–64. <a href="https://doi.org/10.1038/nature11967">https://doi.org/10.1038/nature11967</a></div>
<br>

## Software

<div class="csl-entry">[2] Newville, M., Stensitzki, T., Allen, D. B., &#38; Ingargiola, A. (2014). <i>LMFIT: Non-Linear Least-Square Minimization and Curve-Fitting for Python</i>. <a href="https://doi.org/10.5281/ZENODO.11813">https://doi.org/10.5281/ZENODO.11813</a></div>
<br>

<div class="csl-entry">[3] Ulloa, J. S., Haupert, S., Latorre, J. F., Aubin, T., &#38; Sueur, J. (2021). scikit-maad: An open-source and modular toolbox for quantitative soundscape analysis in Python. <i>Methods in Ecology and Evolution</i>, <i>12</i>(12), 2334–2340. <a href="https://doi.org/10.1111/2041-210X.13711">https://doi.org/10.1111/2041-210X.13711 Dataset </a> </div>
<br>

<div class="csl-entry">[4] McFee, B., Raffel, C., Liang, D., Ellis, D. P., McVicar, M., Battenberg, E., & Nieto, O. &#38; (2015). librosa: Audio and music signal analysis in python. <i>  In Proceedings of the 14th python in science conference </i>, <i>12</i>(12), (Vol. 8). <a href="https://librosa.org/doc/latest/index.html">Librosa</a> </div>

## Audios Dataset

<div class="csl-entry">[5] Xeno-canto Foundation and Naturalis Biodiversity Center &#38; (2005). <a href="https://xeno-canto.org/">xeno-canto:</a>     <i> Sharing bird sounds from around the world.</i> <i>  <a href="https://xeno-canto.org/set/8103">Dissertation Audios Dataset </a> </i></div>
<br>

<div class="csl-entry">[6] Ther Cornell Lab of Ornithology &#38; (2005). ebird <i>, <a href="https://ebird.org/">ebird.com</a> </div>

    
