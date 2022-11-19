<a href="https://github.com/saguileran/birdsongs/"><img src="/docs/img/logo.png" alt="drawing" width="500"/>

# birdsongs

A python package to analyze, visualize and create synthetic syllables.


[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/saguileran/birdsongs/main?labpath=BirdSongs.ipynb)


#  Table of Contents

- [birdsongs](#birdsongs)
- [Table of Contents](#table-of-contents)
  - [- Audios Dataset](#--audios-dataset)
- [Installation](#installation)
  - [Requirments](#requirments)
  - [Downloading](#downloading)
  - [Use](#use)
    - [Define objects](#define-objects)
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

syllables is implemented in python 3.8: It requires:

- numpy
- matplotlib 
- scipy
- peakutils
- lmfit
- librosa
- scikit-maad
- sympy
- pandas
- sklearn
- IPython
- mpl_interactions
- mpl_point_clicker
- playsound
- soundfile

    
## Downloading

In order to use birdsongs, clone the repository and enter to the folder repository

```bat
git clone https://github.com/saguileran/birdsongs.git
cd birdsongs
```
you can verify the current branch with the command `git branch -a`. You have to be in `main` branch, to change the branch use the command `git checkout main`.

The next step is to install the required packages, any of the following commands line will work

```bat
pip install -r ./requirements.txt
python -m pip install -r ./requirements.txt
cd birdsongs
```

the last step is to enter at the birdsongs folder to use its function with the examples audios.


## Use

### Define objects

Import the package as bs

```python
import birdsongs as bs
from birdsongs.utils import * 
```

define a ploter and paths objects, optionally you can specify the audio folder or not 

```python
# audios = "path\to\audios"
# root = "path\to\audios"
# bird_name = "path\to\audios"

ploter = bs.Ploter()  # save = True to save the figures
paths  = bs.Paths()   # root, audios_path, bird_name
```
define and plot the audio birdsong 

```python
bird = bs.Song(paths, no_file=3) # tlim=(t0,tend) you can also give a time interval or frequency limits flim=(f0,fmax)
ploter.Plot(bird)
```
you can also define the time intervals of interest from the whole song with the commands
    
```python
klicker = ploter.FindTimes(bird) # FF_on=True enables fundamental frequency plot
plt.show()
``` 
after close the matplotlib windows run 
    
```python
time_intervals = Positions(klicker)
``` 
The final step is to define the optimizer object to generate the synthetic syllable (song). 
    
```python
syllable  = bs.Syllable(bird)               # you can also give a time or frequency interval flim=(fmin,fmax), tlim=(t0,tend)

brute     = {'method':'brute', 'Ns':11}     # optimization method and its parameters
optimizer = bs.Optimizer(syllable, brute)   # define optimizer to the syllable object

optimizer.optimal_gamma                     # find teh optimal gamma over the whole bird syllables
optimizer.OptimalParams(obj, Ns=11)         # find optimal alpha and beta parameters
    
Display(obj.p)                              # display optimal problem parameters
obj_synth_optimal = obj.Solve(obj.p)        # generate the synthetic syllable with the optimal parameters set
    
ploter.Syllables(obj, obj_synth_optimal)    # plot real and synthetic songs, sound waves and spectrograms
ploter.PlotAlphaBeta(obj_synth_optimal)     # plot alpha and beta parameters in function of time (just syllable has this attributes)
ploter.Result(obj, obj_synth_optimal)       # plot the spectrograms, scores and features of both objects, the real and synthetic
    
bird.WriteAudio();  synth_bird.WriteAudio() # write both objects, real and synthetic
```

other available option is to split the audio with the time intervals define and find the synthetic song

```python
brute          = {'method':'brute', 'Ns':11}
optimizer_bird = bs.Optimizer(bird, method_kwargs=brute)
synth_bird     = optimizer_bird.SongByTimes(time_intervals)
```
    
to visualize and write audio
    
```python
ploter.Plot(synth_bird)
bird.WriteAudio();  synth_bird.WriteAudio()
```
    
    
The repository has some audio examples, in examples/audios folder. You can download and store your own audios in the same folder, the audios must be in WAV format or birdosngs will not import them, we suggest use [audacity](https://www.audacityteam.org/) to convert the audios.

    
<!---
and then add to python 

```bat
pip install -e birdsongs
python -m pip install -e birdsongs
```
-->

# Overview



# Objective

Design, development, and evaluation of a physical model for generating synthetic birdsongs from recorded birdsongs

# Repository Contents

This repository have the documentation, scripts, and results delelop to achive the proposed objective.

The model used, Motor Gestures [1], have been developed by profesog G. Mindlin at the [Dynamical Systems Laboratory](http://www.lsd.df.uba.ar/) (in spanish LSD) of the university of Buenos Aires, Argentina. 

## Physical Model 

Schematic visualization of the complete Motod Gestures model: syrinx, trachea, glotis, and OEC. 

<p align="center"> <img src="./docs/img/model.png" width="700" title="model"></p>

## Programming Object Oriented (POO)

Taking advantege of the POO programming paradig, to avoid repeat long codes execution and easy reproducibility, five objects ared created:

- **Syllable**: define a object from audio syllable with its tempo and spectral features
- **Optimizer**: define a object to optimize function from method and syllables
- **Song**: define a object to read and split an audio song in syllables 
- **Plot**: define a object to plot real and synthec syllables or songs
- **Paths**: define a object to organize folders location

In order to understand the diagram methodology, the following icons will be used. 

<p align="center">  <img src="./docs/img/objects.png" width="500" alt="methodology"></p>

## Implementation

Using the previous objects defined, the optimization problem is solved by following the next diagram 

<p align="center">  <img src="./docs/img/methodology.png" width="600" alt="methodology"></p>

# Results

# References

## Literature

<div class="csl-entry">[1] Amador, A., Perl, Y. S., Mindlin, G. B., &#38; Margoliash, D. (2013). Elemental gesture dynamics are encoded by song premotor cortical neurons. <i>Nature 2013 495:7439</i>, <i>495</i>(7439), 59–64. <a href="https://doi.org/10.1038/nature11967">https://doi.org/10.1038/nature11967</a></div>
<br>

## Software

<div class="csl-entry">[2] Newville, M., Stensitzki, T., Allen, D. B., &#38; Ingargiola, A. (2014). <i>LMFIT: Non-Linear Least-Square Minimization and Curve-Fitting for Python</i>. <a href="https://doi.org/10.5281/ZENODO.11813">https://doi.org/10.5281/ZENODO.11813</a></div>
<br>

<div class="csl-entry">[3] Ulloa, J. S., Haupert, S., Latorre, J. F., Aubin, T., &#38; Sueur, J. (2021). scikit-maad: An open-source and modular toolbox for quantitative soundscape analysis in Python. <i>Methods in Ecology and Evolution</i>, <i>12</i>(12), 2334–2340. https://doi.org/10.1111/2041-210X.13711</div>
<br>

<div class="csl-entry">[4] McFee, B., Raffel, C., Liang, D., Ellis, D. P., McVicar, M., Battenberg, E., & Nieto, O. &#38; (2015). librosa: Audio and music signal analysis in python. <i>  In Proceedings of the 14th python in science conference </i>, <i>12</i>(12), (Vol. 8). https://librosa.org/doc/latest/index.html</div>

## Audios Dataset

<div class="csl-entry">[5] Xeno-canto Foundation and Naturalis Biodiversity Center &#38; (2005). xeno-canto:  https://xeno-canto.org/  <i> Sharing bird sounds from around the world.</i> <i> Dissertation Audios Dataset https://xeno-canto.org/set/8103</i></div>

<div class="csl-entry">[6] Ther Cornell Lab of Ornithology &#38; (2005). ebird <i>,  https://ebird.org/</div>
