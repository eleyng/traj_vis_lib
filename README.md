# traj_vis_lib
Handy visualization scripts for trajectories

# Usage:
Install from source this [repo](https://github.com/eleyng/table-carrying-ai) for dependencies. Trajectories in physical space of .npz format which are of B x T x F (batch, time, feature) format can be leverage these visualization scripts with proper directory & file setup.

# Visualizations 
- playback_traj.py : playback the trajectory, saving images from each frame (and also make video if desired)
- plot_inter_f.py : plot interaction forces over trajectory
- traj_overlay.py : overlay trajectory. first state is more transparent than latest state.

