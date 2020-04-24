# DDPG Bipedal
Code originally from https://github.com/tensorflow/agents/tree/master/tf_agents/agents/ddpg/examples   
Citation can be found at the bottom.

[Video of walker trained with CS5Gamma](/mr/Results/CS5Gamma/videos/390000/openaigym.video.11.699876.video000000.mp4)
## Installation
Before trying to run DDPG Bipedal, please install the following dependencies.
```
pip install python==3.7.4
pip install tensorflow==2.0.0
pip install tensorflow-probability==0.8.0
pip install gym==0.15.4
# Make sure you have swig and pystan packages installed
pip install box2d-py==2.3.8   # Install gym and box2d to get BipedalWalker-v2
sudo apt-get install ffmpeg==1.4   # Install ffmpeg for video creation
pip install tf-agents==0.3.0    # Install TF-Agents for dependencies
pip install tensorboard==2.0.1    # Install Tensorboard or viewing results
pip install jupyterlab==1.2.3     # Install jupyterlab for running .ipynb
```
## Execution

There are two alternatives for running DDPG Bipedal.

To run DDPG_Bipedal.ipynb:
- Download DDPG_Bipedal.ipynb or clone this repository,
- Open jupyter lab and move to location of DDPG_Bipedal.ipynb,
- Open DDPG_Bipedal.ipynb,
- Follow instructions in first cell of DDPG_Bipedal.ipynb.

To run DDPG_Bipedal.py:
- Download DDPG_Bipedal.py or clone this repository,
- Open terminal and change current directory to where DDPG_Bipedal.py is,
- Run ```python DDPG_Bipedal.py --help```,
- Run with arguments according to help description.

## Output Management

DDPG Bipedal will create three subdirectories to the directory the path was set to:
- ```eval``` contains data on policy evaluation,
- ```train``` contains data on losses and training,
- ```vid``` contains videos of trained walker.

You can view training progress while the program is running.

To view policy evaluation data or data on losses and training, follow these instructions:

- Open a terminal, change directory to your path directory, then run:  
```tensorboard --logdir=eval/run_id    # Enter actual run_id at run_id```  
to view evaluation of the current policy.
- For viewing training data, run:  
```tensorboard --logdir=train/run_id    # Enter actual run_id at run_id```.

Then open any browser, and go to http://localhost:6006/ to view graphs of your data.

The vid directory has subdirectories named after the iteration the video was created at.  
Each subdirectory contains one video of the policy being executed.

## TODOs

- Add remaining Hyperparameter Tuning Results.
- Adjust bar to create video. parse hurdle?
- Fill Output_Test with output data

## Citation
```
@misc{TFAgents,
  title = {{TF-Agents}: A library for Reinforcement Learning in TensorFlow},
  author = "{Sergio Guadarrama, Anoop Korattikara, Oscar Ramirez,
    Pablo Castro, Ethan Holly, Sam Fishman, Ke Wang, Ekaterina Gonina, Neal Wu,
    Efi Kokiopoulou, Luciano Sbaiz, Jamie Smith, Gábor Bartók, Jesse Berent,
    Chris Harris, Vincent Vanhoucke, Eugene Brevdo}",
  howpublished = {\url{https://github.com/tensorflow/agents}},
  url = "https://github.com/tensorflow/agents",
  year = 2018,
  note = "[Online; accessed 17.12.2019]"
}
```
