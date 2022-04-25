# Advection-aware autoencoders and long-short-term memory networks for reduced order modeling of parametric, advection-dominated PDEs

This is supporting code for the article
```
Dutta, S.; Rivera-Casillas, P.; Styles, B.; Farthing, M.W.
Reduced Order Modeling Using Advection-Aware Autoencoders.
Math. Comput. Appl. 2022, 27, 34. https://doi.org/10.3390/mca27030034
```
This article is part of the Special Issue: "Computational Methods for Coupled Problems in Science and Engineering".

Email: sourav.dutta@erdc.dren.mil for any questions/feedback.

Advection-aware Autoencoder Architecture
:-----:
<p align="center">
    <img align = 'center' height="500" src="figures/aa_autoencoder_arch_new.jpeg?raw=true">
</p>


## Getting Started

* Generate the high-fidelity snapshot data for the 2D linear advection example by running the script `examples/2DLinearAdvection.py`. It automatically saves the snapshot files in the `data` directory.
* Generate the high-fidelity snapshot data for the 1D Burgers example by running the notebook `examples/1DBurgers_data.ipynb`. It automatically saves the snapshot files in the `data` directory and generates snapshot visualizations.

### Dependencies

* Python 3.x
* Tensorflow TF 2.x. Install either the CPU or the GPU version depending on available resources.
* A list of all the dependencies are provided in the [requirements](requirements.txt) file.

### Executing program

* The AA autoencoder training and evaluation can be performed using the notebooks `examples/AA_autoencoder_parametric_2DLinearAdvection.ipynb` and `examples/AA_autoencoder_parametric_1DBurgers.ipynb`.
* The performance of the various AA autoencoder models are compared in the notebooks `examples/AA_autoencoder_comparison_2DLinearAdvection.ipynb` and `examples/AA_autoencoder_comparison_1DBurgers.ipynb`.
* The LSTM and parametric LSTM models for the 1D Burgers' example are trained and evaluated using the notebooks `examples/LSTM_1DBurgers.ipynb` and `examples/pLSTM_parametric_1DBurgers.ipynb`.
