In this project, I employed stochastic models (Monte Carlo methods, Hidden Markov Models) for parameter learning from the data with application to 
physical and biological sciences. I built a Bayesian non-parametric model for counting number of molecules on top of the cell by analyzing images from 
super-resolution microscopy.

SMLM is used to visualize small biological structures. In SMLM, all proteins are labeled with photo activatable fluorescent proteins (PA-FPs) and are 
expressed in their native amount. When they are exposed to light they photo-convert and emit lights stochatically through time. Then by repeatedly 
imaging a small random subset of fluorescent molecules in the sample, images with sparse support can be created and thereby allows extremely high 
accuracy in determining the locations of the molecules. I first built a Bayesian nonparametric model for this purpose. The method worked fine for high 
signal to noise ratios cases. However, the method was underestimating the number of molecules under challenging signal-to-noise conditions and high 
emitter densities.
