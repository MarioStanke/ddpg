# Ideas to try

1)  According to Silver et al. 2014 the policy gradient is an expectation wrt to the state distribution rho.
   Therefore, in the replay memory states sampled at time t should be chosen with a probability discounted by gamma^t

2) Increase the time step during training: repeat the action computed by the policy for s>1 time steps.
  This can be done with a slight modification of the agent by adding a counter.
  If s=2 is better than s=1 it is a sign that a method looking ahead further would be even better.

3) Derive the math for the policy gradient if s>1 steps are done with the deterinistic policy. How can tensorflow compute the derivative of a compound policy, e.g. letting the original policy act twice pi(env(pi))?

4) Include the previous state in the input to the critic network, i.e. Q=Q(s_t-1, s_t, a_t)
and/or the previous action to the actor network.  The difference between state features (like x-pos) could be more useful, e.g. to use velocities. The actions could be smoother.


  
