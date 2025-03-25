NEURON model files derived from the paper: 
Contribution of the Kv3.1 potassium channel to high-frequency firing in mouse auditory neurones
Lu-Yang Wang, Li Gan, Ian D. Forsythe and Leonard K. Kaczmarek
J. Physiol (1998), 501.9, pp. 183-194

This model will generate graphs similar to those in figure 5.  For
example one could compare the 400 Hz firing with and without Kv3.1.
The Kv3.1 current can be turned off by setting the value of gbar_HT to 0.


To Compile in UNIX:
++++++++++++++++++++++++

1. After unzipping the model file, run nrnivmodl

2. Start the program with nrngui mosinit.hoc


To Compile in Windows:
++++++++++++++++++++++++

1. Run the mknrndll DOS box icon in the Neuron directory on the start menu

2. Navigate to the directory containing the mod files

3. Type in: mknrndll

4. Run mosinit.hoc
