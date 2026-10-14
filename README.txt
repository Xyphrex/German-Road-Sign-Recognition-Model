model.py will run on cpu but is much faster on gpu. Tutorials for using a gpu
with tensorflow vary based on system architecture. Since I was using windows
I used WSL2 along with miniconda to setup an environment which contained the
appropriate pip libraries and nvidia libraries

using "pip install tensorflow[with-gpu]" worked to install all the correct
libraries for me. after that having the Folders "GTSRB Test" and "GTSRB Train"
containing the corresponding test and train data are expected for the model to
run. To run the model simply type "python model.py" after installing all the
required libraries.

You must download and place GTSRB Test and GTSRB Train into this folder
they may not be named correctly when downloaded from the link in the assignment
spec .pdf and will need to be renamed. They were not zipped for submission
as it would have been too large and taken too long to submit to canvas.