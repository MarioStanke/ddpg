# DDPG
Code originally from https://github.com/tensorflow/agents/tree/master/tf_agents/agents/ddpg/examples

Citation below
## Installation and Execution
```
pip install gym==0.15.4
pip install box2d-py==2.3.8   # Install gym to get BipedalWalker-v2
sudo apt-get install ffmpeg    # Install ffmpeg for video creation
pip install tf-agents==0.3.0    # Install TF-Agents for dependencies
pip install tensorboard    # Install Tensorboard or viewing results
pip install jupyterlab     # Install jupyterlab for running scripts

# Open DDPG_Bipedal.ipynb script with jupyter lab
# Follow instructions in first cell, then run all cells

# Alternatively, get DDPG_Bipedal.py, open terminal 
# Then change current directory to where DDPG_Bipedal.py is
# Run 'python DDPG_Bipedal.py --help' and follow instructions to set relevant parameters.
```
## TODOs
```
- Add remaining Hyperparameter Tuning Results.
```
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
