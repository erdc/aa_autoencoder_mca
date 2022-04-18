# Advection-aware autoencoders and long-short-term memory networks for reduced order modeling of parametric, advection-dominated PDEs

This is supporting code for the manuscript "Reduced order modeling using advection-aware autoencoders" by Sourav Dutta, Peter Rivera-Casillas, Brent Styles and Matthew Farthing, that has been *accepted* for publication in the *Mathematical and Computational Applications* journal for the Special Issue: "Computational Methods for Coupled Problems in Science and Engineering". The AA autoencoder model is tested with a parametric linear advection problem and a parametric viscous advecting shock problem. 

Email: sourav.dutta@erdc.dren.mil for any questions/feedback.

Advection-aware Autoencoder Architecture
:-----:
<p align="center">
    <img align = 'center' height="500" src="figures/aa_autoencoder_arch_new.jpg?raw=true">
</p>


## Getting Started


### Dependencies

* Python 3.x
* Tensorflow TF 2.x. Install either the CPU or the GPU version depending on available resources.

A list of all the package requirements along with version information will be provided in the [requirements](requirements.txt) file.

### Executing program

* Generate the high-fidelity snapshot data by running the script `examples/2DLinearAdvection.py`. It automatically saves the snapshot files in the `data` directory.
* The AA autoencoder training and evaluation can be performed using the notebook `examples/AA_autoencoder_parametric_2DLinearAdvection.ipynb`.
