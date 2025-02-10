<h1>This repository is migrating to <a href="https://github.com/wavesongs/wavesongs">WaveSongs</a> </h1>
<a href="https://github.com/saguileran/birdsongs/"><img src="./assets/img/logo.png" width="500"></a>

# Birdsongs

A python package for analyzing, visualizing and generating synthetic birdsongs from recorded audios.


[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/saguileran/birdsongs/main?labpath=BirdSongs.ipynb)

---

##  Table of Contents
- [Birdsongs](#birdsongs)
  - [Table of Contents](#table-of-contents)
  - [Objective](#objective)
  - [Overview](#overview)
  - [Repository Contents](#repository-contents)
  - [Python Implementation](#python-implementation)
    - [Physical Model](#physical-model)
    - [Object-Oriented Thinking](#object-oriented-thinking)
  - [Methodology](#methodology)
  - [Installation](#installation)
    - [Requirments](#requirments)
    - [Download](#download)
  - [Use](#use)
    - [Import Libraries](#import-libraries)
    - [Define Objects](#define-objects)
      - [Path and Plotter](#path-and-plotter)
      - [Birdsong](#birdsong)
      - [Syllable](#syllable)
    - [Solve](#solve)
    - [Visualize](#visualize)
  - [Results](#results)
  - [Applications](#applications)
  - [References](#references)
    - [Literature](#literature)
    - [Software](#software)
    - [Audios Dataset](#audios-dataset)
---
  
  
## Objective

Design, development, and evaluation of a computational-physical model for generating synthetic birdsongs from recorded samples.

## Overview

Study and Python implementation of the **motor gestures for birdsongs** model created by Prof. [G Mindlin](https://scholar.google.com.ar/citations?user=gMzZPngAAAAJ&hl=en). This model explains the physics of birdsong by simulating the organs involved in sound production in birds, including the syrinx, trachea, glottis, oro-esophageal cavity (OEC), and beak, using ordinary differential equations (ODEs).

This work presents an automated model for generating synthetic birdsongs that are comparable to real birdsong in both spectrographic and temporal aspects. The model utilizes motor gestures for birdsongs model and an audio recording of real birdsong as input. Automation is achieved by formulating a minimization problem with three control parameters: air sac pressure of the bird’s bronchi, labial tension of the syrinx walls, and a time scale constant. This optimization problem is solved using numerical methods, signal processing tools, and numerical optimization techniques. The objective function is based on the Fundamental Frequency (also called pitch, denoted as FF or F0) and the Spectral Content Index (SCI) of both synthetic and real syllables.

The model is tested and evaluated on three different Colombian bird species: Zonotrichia capensis, Ocellated Tapaculo, and Mimus gilvus, using recorded samples downloaded from [Xeno-Canto](https://xeno-canto.org/) and [eBird](https://ebird.org/home) audio libraries. The results show relative errors in FF and SCI of less than 10%, with comparable spectral harmonics in terms of number and frequency, as detailed in the [Results](#results) section.


## Repository Contents

This repository contains the documentation, scripts, and results developed to achive the proposed objective. The files and information are divided in branches as follows:

- **main:** Python package with model code implementation, tutorial examples, example data, and results.
- **dissertation:** Latex document of the bachelor's dissertation: *Design, development, and evaluation of a computational physical model to generate synthetic birdsongs from recorded samples*.
- **gh-pages:** Archieves for the [BirdSongs](https://saguileran.github.io/birdsongs/) website, a more in-depth description of the package.
- **results:** Some results obtanied from the tutorial examples: image, audios and motor gesture parameters (.csv).

The physical model used, Motor Gestures for birdsongs [1], have been developed by profesog G. Mindlin at the [Dynamical Systems Laboratory](http://www.lsd.df.uba.ar/) (in spanish LSD) of the university of Buenos Aires, Argentina. 

## Python Implementation 

### Physical Model 

Schematic description of the physical model **motor gestures for birdsongs** with the organs involved in the sound production (syrinx, trachea, glotis, OEC, and beak) and their corresponding ODEs. 

<figure><center>
  <img src="./assets/img/model.png" width="600" alt="methodology">
  <figcaption><b>Figure 1.</b>  Motor gestures model diagram.</figcaption>
</figure></center>

### Object-Oriented Thinking

By leveraging the Object-Oriented Programming (OOP) paradigm, the need for lengthy code is minimized. Additionally, the execution and implementation of the model are efficient and straightforward, allowing for the creation and comparison of several syllables with a single line of code. To solve the optimization problem and to analyze and compare real and synthetic birdsong, five objects are created:

- **BirdSong**: Read an audio using its file name and a path sobject, it computes the audio spectral and temporal features. It can also split the audio into syllables (in process).
- **Syllable**: Create a birdsong syllable from a birdsong object using a time interval that can be selected in the plot or defined as a list. The spectral and temporal features of the syllable are automatically computed. 
- **Optimizer**: Create an optimizer that solves the minimization problem using the method entered (the default is brute force but can be changed to leastsq, bfgs, newton, etc. The use of a different method to brute force need to add additional parameters. Further information in [lmfit](https://lmfit.github.io/lmfit-py/fitting.html)) in a feasible region that can be modified. 
- **Plotter**: Visualize the birdsong and sylalble objects and their spectral and temporal features. It also include a functionality to select points from the spectrum.
- **Paths**: Manage the package paths, audio files and results directories.

For each object an icon is defined as follows:

<figure><center>
  <img src="./assets/img/objects.png" width="600" alt="methodology">
  <figcaption><b>Figure 2.</b> Objects implemented.</figcaption>
</figure></center>


This approach simplifies the interpretation of the methodology diagram. Each icon represents an object that handles different tasks. The major advantage of this implementation is the ability to easily compare features between syllable or chunk (small part of a syllable) objects.

## Methodology

Using the previous defined objects, the optimization problem is solved by following the next steps below:

<!-- <p align="center"> <img src="./assets/img/methodology.png" width="600" alt="methodology"></p>
-->
<figure><center>
  <img src="./assets/img/methodology.png" width="600" alt="methodology">
  <figcaption><b>Figure 3.</b> Methodology diagram.</figcaption>
</figure></center>


Each step includes the icon of the object involved. The final output is a parameters object (a data frame similar to the lmfit library parameters objects) containing the optimal control parameter coefficients for the motor gestures that best reproduce the real birdsong.

## Installation

### Requirments

`birdsong` is implemented in Python 3.8 but is also tested in Python 3.10 and 3.11. The required packages are listed in the file [requirements.txt](https://github.com/saguileran/birdsongs/blob/main/requirements.txt).
    
### Download

To use **birdsongs**, clone the `main` branch of the repository and go to its root folder.

```bat
git clone -b main --single-branch https://github.com/saguileran/birdsongs.git
cd birdsongs
```
You can clone the whole repository using the code
`git clone https://github.com/saguileran/birdsongs.git` but since it is very large only the main branch is enough. To change the branch use the command `git checkout ` follow of the branch name of interest.

The next step is to install the required packages, any of the following commands lines will work:

```bat
pip install -r ./requirements.txt
python -m pip install -r ./requirements.txt
```

If you are using a version of Python higher than 3.10, to listening the audios you must execute


```bat
pip install playsound@git+https://github.com/taconi/playsound
```

<!--
You can now use the package in a python terminal opened at the birdsongs folder. 

To use the package from any folder install the repository, this can be done with any of the two following lines
-->

Now, install the birdsong package.

```bat
python .\setup.py install
```

or using pip, any of the following command lines should work:

```bat
pip install -e .
pip install .
```

That's all. Now let's create a synthetic birdsong!
   
<!---
and then add to python 

```bat
pip install -e birdsongs
python -m pip install -e birdsongs
```
-->

Take a look at the tutorials notebooks for basic uses: physical model implementation, [motor-gestures.ipynb](./tutorials/motor-gestures.ipynb); define and generate a syllable from a recorded birdsong, [syllable.ipynb](./tutorials/syllable.ipynb); or to generate a whole birdsong, several syllables, [birdsong.ipynb](./tutorials/birdsong.ipynb),

## Use

### Import Libraries

Import the birdonsg package as `bs` 

```python
import birdsongs as bs
from birdsongs.utils import *
```  

### Define Objects

#### Path and Plotter

First, define the plotter and paths objects, optionally you can specify the audio folder or enable plotter to save figures 

```python
root    = "root\path\files"     # default ..\examples\
audios  = "path\to\audios"      # default ..\examples\audios\
results = "path\to\results"     # default ..\examples\results\

paths  = bs.Paths(root, audios, results)
plotter = bs.Ploter(save=True)  # images are saved at ./examples/results/Images/
```

Displays the audios file names found with the `paths.AudiosFiles(True)` function, if the folder has a *spreadsheet.csv* file this function displays all the information about the files inside the folder otherwise it diplays the audio files names found.

#### Birdsong
  
Define and plot the wave sound and spectrogram of the sample *XC11293*. You can use both mp3 and wav files but in Windows maybe you can get errors from `librosa.load`. 

```python
birdsong = bs.BirdSong(paths, file_id="XC11293", NN=512, umbral_FF=1., Nt=500,
                       tlim=(0,60), flim=(100,20e3) # other features
                      )
plotter.Plot(birdsong, FF_on=False)  # plot the wave sound and spectrogram without FF
birdsong.Play()                     # listen the birdsong
```

>[!NOTE]
>The parameter *Nt* is related to the envelope of the waveform, for long audios large Nt while for short audios small Nt. 

#### Syllable
  
Define the syllable using a time interval of interest and the previous birdsong object. The syllable inherits the birdsong attributes (NN, flim, paths, etc). To select the two time interval points (start and end of the syllable) from the plot change the `SelectTime_on` argument of the `plotter.Plot()` funtion to `True`.
    
```python
# selec time intervals from the birdsong plot, you can select a single pair
plotter.Plot(birdsong, FF_on=False, SelectTime_on=True) # select 
```

Then, define a birdsong syllable with the time interval selected

```python
time_intervals = Positions(plotter.klicker)             # save
print(time_intervals)                                  # display

# define the syllable object
syllable = bs.Syllable(birdsong, tlim=time_intervals[0], Nt=30, ide="syllable")
plotter.Plot(syllable, FF_on=True)
``` 

>[!IMPORTANT]
>The algorithm used to calculate the fundamental frequency does not perform well at the extremes of the syllable. To avoid issues, do not select the exact extremes; instead, choose a slightly shorter segment of the syllable.
  
### Solve
  
Now let's define the optimizer object to generate the synthetic syllable by solving the optimization problem. First, create the optimizer object by specifying the optimization method, its parameters, and the syllable of interest.

```python
brute_kwargs = {'method':'brute', 'Ns':11}          # optimization method,  Ns is the number of grid points
optimizer    = bs.Optimizer(syllable, brute_kwargs) # optimizer object
```
Then, execute the solver to find the optimal time scalar constant and the optimal motor gesture parameters (labial tension and air sac pressure vairables) 
```python
optimal_gm   = optimizer.OptimalGamma(syllable)     # find optimal gamma (time scale constant) 
optimizer.OptimalParams(syllable, Ns=11)            # find optimal parameters coefficients
#syllable, synth_syllable = optimizer.SongByTimes(time_intervals)   # find optimal parameters over several time intervals
```
    
Finally, define the synthetic syllable object with the optimal values found above.

```python
synth_syllable = syllable.Solve(syllable.p)
```

### Visualize
  
Finally, visualize and write the optimal synthetic audio. 
    
```python
plotter.Plot(synth_syllable);                # sound wave and spectrogram of the synthetic syllable
plotter.PlotVs(synth_syllable);              # physical model variables over the time
plotter.PlotAlphaBeta(synth_syllable);       # motor gesture curve in the parametric space
plotter.Syllables(syllable, synth_syllable); # synthetic and real syllables spectrograms and waveforms
plotter.Result(syllable, synth_syllable);    # scoring variables and other spectral features

birdsong.WriteAudio();  synth_syllable.WriteAudio(); # write both audios at ./examples/results/Audios
```
  
>[!NOTE]
> 
>To generate a single synthetic syllable (or chunck) you must have defined a birdsong (or syllable) and the process is as follows:
>
>1. Define a path object.
>2. Define a birdsong object using the above path object, it requeries the audio file id. You can also enter the length of the window FFT and the umbral (threshold) for computing the FF, between others. 
>3. Select or define the time intervals of interest.
>4. Define an optimization object with a dictionary of the method name and its parameters.
>5. Find the optimal gammas for all the time intervals, or a single, and average them.
>6. Find and export the optimal labia parameters for each syllable, the motor gesture curve.
>7. Generate synthetic birdsong from the previous control parameters found.
>8. Visualize and save all the syrinx, scoring, and result variables.
>9. Save both synthetic and real syllable audios.

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
    
plotter.Syllables(obj, obj_synth_optimal)    # plot real and synthetic songs, sound waves and spectrograms
plotter.PlotAlphaBeta(obj_synth_optimal)     # plot alpha and beta parameters in function of time (just syllable has this attributes)
plotter.Result(obj, obj_synth_optimal)       # plot the spectrograms, scores and features of both objects, the real and synthetic
    
bird.WriteAudio();  synth_bird.WriteAudio() # write both objects, real and synthetic
```
-->
    
The repository has some audio examples, in the [./examples/audios](https://github.com/saguileran/birdsongs/tree/main/examples/audios) folder. You can download and store your own audios in the same folder or enter the audio folder path to the Paths object. 
<!--
The package also has a function to download audios from Xeno-Canto: birdsong.util.DownloadXenoCanto().
-->
The audios can be in WAV of MP3 format. If you prefer WAV format, we suggest use [Audacity](https://www.audacityteam.org/) to convert the audios without any issue.

 
## Results

The model is tested and evaluated with different syllables of the birdsong of the Rufous Collared Sparrow. Results are located at [examples/examples](./examples/results), images and audios. For more information visit the project website [birdsongs](https://saguileran.github.io/birdsongs/) or access the [results](https://saguileran.github.io/birdsongs/results/) page. 

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

The PDF document of the bachelor thesis, <a href="https://github.com/saguileran/birdsongs/blob/dissertation/dissertation.pdf">Design, development, and evaluation of a computational physical model to generate synthetic birdsongs from recorded samples</a>, is stored in the `dissertation` brach of this repository.

<figure><center>
  <img src="https://raw.githubusercontent.com/saguileran/birdsongs/gh-pages/assets/img/cover.jpg" width="250" height="350"><img src="https://raw.githubusercontent.com/saguileran/birdsongs/gh-pages/assets/img/under-cover.png" width="250" height="350">
  <figcaption><b>Figure 6.</b> Bachelor's thesis PDF document.</figcaption>
</figure></center>

## Applications

Some of the applications of this model are:

- **Data Augmentation**: use the model to create numerous synthetic syllables, it can be done by creating a syntetic birdsong and then varying its motor gestures parameters to get similar birdsongs. 
- **Birdsongs Descriptions**: characterize and compare birdsongs using the motor gestures parameters.


## References

### Literature

<div class="csl-entry">[1] Amador, A., Perl, Y. S., Mindlin, G. B., &#38; Margoliash, D. (2013). Elemental gesture dynamics are encoded by song premotor cortical neurons. <i>Nature 2013 495:7439</i>, <i>495</i>(7439), 59–64. <a href="https://doi.org/10.1038/nature11967">https://doi.org/10.1038/nature11967</a>.</div>
<br>

### Software

<div class="csl-entry">[2] Newville, M., Stensitzki, T., Allen, D. B., &#38; Ingargiola, A. (2014). <i>LMFIT: Non-Linear Least-Square Minimization and Curve-Fitting for Python</i>. <a href="https://doi.org/10.5281/ZENODO.11813">https://doi.org/10.5281/ZENODO.11813</a>.</div>
<br>

<div class="csl-entry">[3] Ulloa, J. S., Haupert, S., Latorre, J. F., Aubin, T., &#38; Sueur, J. (2021). scikit-maad: An open-source and modular toolbox for quantitative soundscape analysis in Python. <i>Methods in Ecology and Evolution</i>, <i>12</i>(12), 2334–2340. <a href="https://doi.org/10.1111/2041-210X.13711">https://doi.org/10.1111/2041-210X.13711 Dataset</a>.</div>
<br>

<div class="csl-entry">[4] McFee, B., Raffel, C., Liang, D., Ellis, D. P., McVicar, M., Battenberg, E., & Nieto, O. &#38; (2015). librosa: Audio and music signal analysis in python. <i>  In Proceedings of the 14th python in science conference </i>, <i>12</i>(12), (Vol. 8). <a href="https://librosa.org/doc/latest/index.html">Librosa</a>. </div>

### Audios Dataset

<div class="csl-entry">[5] Xeno-canto Foundation and Naturalis Biodiversity Center &#38; (2005). <a href="https://xeno-canto.org/">Xeno-Canto:</a> <i> Sharing bird sounds from around the world.</i> <i>  <a href="https://xeno-canto.org/set/8103">Dissertation Audios Dataset</a>.</i></div>
<br>

<div class="csl-entry">[6] Fink, D., T. Auer, A. Johnston, M. Strimas-Mackey, S. Ligocki, O. Robinson, W. Hochachka, L. Jaromczyk, C. Crowley, K. Dunham, A. Stillman, I. Davies, A. Rodewald, V. Ruiz-Gutierrez, C. Wood. 2023. <i>eBird Status and Trends, Data Version: 2022; Released: 2023</i>. Cornell Lab of Ornithology, Ithaca, New York. <a href="https://doi.org/10.2173/ebirdst.2022">https://doi.org/10.2173/ebirdst.2022</a>. </div>
