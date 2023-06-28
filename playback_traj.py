import copy
import sys
import time
from os.path import dirname, join

import numpy as np
import pygame
import torch
import wandb

from cooperative_transport.gym_table.envs.table_env import TableEnv
from cooperative_transport.gym_table.envs.custom_rewards import \
    custom_reward_function
from cooperative_transport.gym_table.envs.utils import (CONST_DT, FPS,
                                                        MAX_FRAMESKIP,
                                                        WINDOW_H, WINDOW_W,
                                                        debug_print,
                                                        get_keys_to_action,
                                                        init_joystick,
                                                        set_action_keyboard)
from libs.planner.planner_utils import (is_safe, pid_single_step, tf2model,
                                        tf2sim, update_queue)
from libs.sim_utils import make_video


# INFO: Some parameters indicating trajectory. if you know the name of the trajectory file you want to visualize, you can just set the path to that file as the value of the variable traj. Must set 2 other params: map_cfg (includes map information specific to the played back trajectory) and map_yml (generic config info from which the map info was sampled to generate the map).

# OPTIONS: plot the trajectory frame by frame, or save a video of the trajectory (comment out make_video())

subject = 2
planner_type = "vrnn"
ep = 602
full_path_to_vid_save = "ep-{}".format(ep)
hact = "False"
traj = "datasets/rnd_obstacle_v3/random_run_name_3/ep_602.npz"
# traj = "results/hil-dp/subject-{}/eval_hil_seed-88_R-planner-{}_H-real-joystick_exp-random_run_name_3_hact-{}_subject-{}/eval_hil_seed-88_R-planner-{}_H-real-joystick_exp-random_run_name_3_hact-{}_subject-{}-ep_{}.npz".format(subject, planner_type, hact, subject, planner_type, hact, subject, ep)
map_cfg = "demo/rnd_obstacle_v3/random_run_name_3/map_cfg/ep_{}.npz".format(ep)
map_yml = "cooperative_transport/gym_table/config/maps/rnd_obstacle_v3.yml" 


def playback_traj(
    env,
    exp_run_mode="replay_traj",
    robot="data",
    planner_type=None,
    playback_trajectory=None,
):

    # -------------------------------------------- CHECK EXPERIMENT ARGS -------------------------------------------- #

    assert playback_trajectory is not None, "Must provide playback trajectory"
    if len(playback_trajectory["actions"].shape) == 3:
        print(playback_trajectory["actions"].shape)
        playback_trajectory["actions"] = playback_trajectory["actions"].squeeze() 
    assert (
        exp_run_mode == "replay_traj", robot == "data"
        ), "Must be in replay_traj mode and robot must be data"

    # ----------------------------------------------------- SIMULTAOR SETUP -----------------------------------------------#

    # reset environment
    obs = env.reset()
    n_iter = 0

    while n_iter < playback_trajectory["states"].shape[0]:
        u_all = playback_trajectory["actions"][n_iter]
        obs, reward, done, info = env.step(u_all)
        
        state = playback_trajectory["states"][n_iter]
        env.table.x = state[0]
        env.table.y = state[1]
        theta = np.arctan2(state[3], state[2]) % (2 * np.pi)
        env.table.angle = theta
        env.redraw()
        env.render(path="RAL23playedback", n_step=n_iter)
        n_iter += 1

    pygame.quit()

    env.close()
    del env

# -------------------------------------------- MAIN -------------------------------------------- #
env = TableEnv(
        render_mode="gui",
        control="joystick",
        map_config=map_yml,
        run_mode="eval",
        load_map=map_cfg,
        ep=ep,
        dt=CONST_DT,
    )
playback_trajectory = dict(np.load(traj, allow_pickle=True))

playback_traj(env, playback_trajectory=playback_trajectory, planner_type=planner_type)
make_video("/home/eleyng/table-carrying-ai/RAL23playedback/img", full_path_to_vid_save)
