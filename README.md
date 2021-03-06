# DDPG Bipedal
Code originally from https://github.com/tensorflow/agents/
Citation can be found at the bottom.

[Video of walker trained with CS5Gamma](/Results/CS5Gamma/videos/390000/openaigym.video.11.699876.video000000.mp4)
## Installation
Before trying to run DDPG Bipedal, please install the following dependencies.
```
pip install python==3.7.4
pip install tensorflow==2.0.0
pip install tensorflow-probability==0.8.0
pip install gym==0.15.4            # Make sure you have swig and pystan packages installed
pip install box2d-py==2.3.8        # Install gym and box2d to get BipedalWalker-v2
sudo apt-get install ffmpeg==1.4   # Install ffmpeg for video creation
pip install tf-agents==0.3.0       # Install TF-Agents for ddpg
pip install tensorboard==2.0.1     # Install Tensorboard for viewing results
pip install jupyterlab==1.2.3      # Install jupyterlab for running .ipynb
```
## Execution

There are two alternatives for running DDPG Bipedal.

To run DDPG_Bipedal.ipynb:
- Clone this repository, change directory to /ddpg,
- Open jupyter lab and DDPG_Bipedal.ipynb,
- Follow instructions in first cell of DDPG_Bipedal.ipynb to run.

To run DDPG_Bipedal.py:
- Clone this repository, change directory to /ddpg,
- Run ```python DDPG_Bipedal.py --help```,
- Run with arguments according to help description.

## Output Management

DDPG Bipedal will create three subdirectories to the directory the path was set to:
- ```eval``` contains data on policy evaluation,
- ```train``` contains data on losses and training,
- ```vid``` contains videos of trained walker.

You can view any data progress while the program is running.  
To view policy evaluation data or data on losses and training,  
move to your path directory, run one of these commands:

- To view evaluation data, run:  
```tensorboard --logdir=eval/run_id    # Enter actual run_id at run_id```  
to view evaluation of the current policy.
- To view training data, run:  
```tensorboard --logdir=train/run_id    # Enter actual run_id at run_id```.

Then open any browser, and go to http://localhost:6006/ to view graphs of your data.

The vid directory has subdirectories named after the iteration the video was created at.  
Each subdirectory contains one video of the policy being executed.  
Videos will only be created if your policy achieves an average return greater than 230.  

See [here](/Output_Example/) for a preview of what your output will look like.

## TODOs

- Add remaining Hyperparameter Tuning Results.
- Adjust bar to create video. parse hurdle?

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
