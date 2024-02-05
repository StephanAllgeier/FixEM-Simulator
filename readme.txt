This module offers the option of generating fixation eye movements (FEM) synthetically.
Two different methods are available for this purpose. The resulting trajectories are saved as a .csv-file with the
columns "t", "x" and "y". In the case of the mathematical model a boolean column with flags for drift and
microsaccades are given as well.

Firstly, a mathematical model based on a random walk can be used, which is based on the algorithm by Nau. et. al
[NAU, M. A.; PLONER, S. B.; MOULT, E. M.; FUJIMOTO, J. G.; MAIER, A. K.: Open Source Simulation of Fixational Eye Drift
Motion in OCT Scans, available at https://github.com/sploner/eye-motion-simulation]. This was extended in this work so
that microsaccades and tremor are now also generated.


Secondly, generation is possible using a previously trained GAN, which is based on the code from Severo et. al
[SEVERO, D.; AMARO, F.; HRUSCHKA, JR, E. R.; COSTA, A. S. D. M.: Ward2ICU: A
Vital Signs Dataset of Inpatients from the General Ward, available at https://github.com/sploner/eye-motion-simulation]. 

The modules are organised as follows:
The "GeneratingTraces_MathematicalModel" folder provides all the modules required to generate FEM based on the
mathematical model. It contains the "RandomWalk.py" file, which provides the interface for operation with the
"RandomWalk()" function.

The "GeneratingTraces_RCGAN" folder provides all the functions for training and generating new trajectories using RCGAN.
This contains functions for trianing, "run_experiment()" and for generating new trajectories using the file
"generate_dataset.py".

Other files included are the label files from the freely accessible datasets of the Roorda Labobarory and the GazeBase
dataset.

The "ProcessingData" folder also contains various functions for processing and visualising the data.


All functions can be called via a Gui. This offers the option of easily generating trajectories. It should be noted that
the use of the Gui is a simplified version of the generation option; not all parameters can be configured there.
The respective function of the module must be used for more precise configuration of the trajectories.
The configuration options offered provide the most necessary parameters.

