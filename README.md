# ddpg
Code originally from https://github.com/udacity/deep-reinforcement-learning
## Installation and Execution
```
pip3 install gym
pip3 install 'gym[box2d]'
sudo apt-get install ffmpeg # to produce videos
cd ms
python3 DDPG.py
```
## Literature
 - [CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING](https://arxiv.org/abs/1509.02971), Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver & Daan Wierstra
 - Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M. (2014, June). [Deterministic policy gradient algorithms](http://www.jmlr.org/proceedings/papers/v32/silver14.pdf).

## Notes
Watch latest video in slow motion
```
ls -rt vid/*/*/*.mp4 | tail -n 1 | xargs mpv --speed=.1
```

## TODOs
 - make sure that exact same outcome is obtained with the same random seed
 
