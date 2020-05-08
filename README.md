## Welcome 

This repository contains some of my projects in Python and R. They vary in subject and scope and cover Bayesian statistics, data engineering, Deep learning, tatistical analysis, machine learning, data acquisition, data cleaning, data exploration, and data visuzalition. If you have any questions or feedback about any of my projects (data sources, inspiration, critiques) please contact me at mahzadkhoshlessan@gmail.com.

### [AI for Energy: Using NLP to Find Barriers to Humanizing Energy Transition](https://github.com/mkhoshle/PORTFOLIO.github.io/tree/master/AI-for-Energy)
The philosophy behind this project is to find ways that governments can benefit from to involve
people in accelerating energy transition and taking advantage of sustainable energy
resources. In fact, people play an important role besides state-of-the-art technologies and
business boosters. How could we get these insights? I used NLP to answer this question.

### [Parallel Analysis of Molecular Dynamic Trajectories](https://github.com/mkhoshle/paper-hpc-py-parallel-mdanalysis)
Paper draft investigating approaches to increase the performance of analysing MD trajectories with Python on HPC resources. 

#### [Supplement Information](https://github.com/mkhoshle/supplement-hpc-py-parallel-mdanalysis)
Supplementary material (scripts and documentation) for the paper Parallel Performance of Molecular Dynamics Trajectory Analysis, available at https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.5789.

### [Molecular Counting & Single Molecule Localization Microscopy in Living Cells Using Deep-Learning](https://github.com/mkhoshle/PORTFOLIO.github.io/tree/master/Molecular%20Counting)
In this project, I employed a deep convolutional neural network to quantify the number of particles in a region of interest. I implemented a pre-trained neural network and parallelized the model on GPU. The deep convolutional neural network was able to overcome important theoretical challenges arises from the complex photo-physics of fluorescent molecules and gain acceptable accuracy up to a density of 6 [emitter/m2].
![output](https://github.com/mkhoshle/PORTFOLIO.github.io/blob/master/Molecular%20Counting/output.png)

### [Bayesian Nonparametric Approach for Reliable Molecular Counting in Living Cells](https://github.com/mkhoshle/PORTFOLIO.github.io/tree/master/Counting-Bayesian-statistics)
In this project, I employed stochastic models (Monte Carlo methods, Hidden Markov Models) for parameter learning from the data with application to physical and biological sciences. I built a Bayesian non-parametric model for counting number of molecules on top of the cell by analyzing images from super-resolution microscopy. 

SMLM is used to visualize small biological structures. In SMLM, all proteins are labeled with photo activatable fluorescent proteins (PA-FPs) and are expressed in their native amount. When they are exposed to light they photo-convert and emit lights stochatically through time. Then by repeatedly imaging a small random subset of fluorescent molecules in the sample, images with sparse support can be created and thereby allows extremely high accuracy in determining the locations of the molecules. I first built a Bayesian nonparametric model for this purpose. The method worked fine for high signal to noise ratios cases. However, the method was underestimating the number of molecules under challenging signal-to-noise conditions and high emitter densities.

### [Clinical Trial Data](https://github.com/mkhoshle/PORTFOLIO.github.io/tree/master/Clinical_trial_data)
A researcher needs your statistical expertise to evaluate an intervention. The six-month intervention was to give subjects a special exercise plan. The researcher randomized 2,500 people to the treatment group (which receives the intervention) and another 2,500 to the control group (does not receive the intervention).  The physician wishes to evaluate the subjects’ health outcomes (weight and self-rated health) before the start of the intervention and immediately after the intervention. The physician wants to know whether the new exercise plan will affect one’s overall health (measured by change in weight and self-rated health) differently for those receiving the new exercise plan versus those not receiving the exercise plan.


