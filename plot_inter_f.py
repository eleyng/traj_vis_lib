import argparse
from os import mkdir
from os.path import isdir, isfile, join

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

from cooperative_transport.gym_table.envs.utils import (WINDOW_H, WINDOW_W,
                                                        obstacle_size, L)
from libs.sim_utils import make_video


def compute_interaction_forces(table_state, f1, f2):
    table_center_to_player1 = np.array(
            [
                table_state[0] + (L/2) * table_state[2],
                table_state[1] + (L/2) * table_state[3],
            ]
        )
    table_center_to_player2 = np.array(
        [
            table_state[0] - (L/2) * table_state[2],
            table_state[1] - (L/2) * table_state[3],
        ]
    )
    inter_f = (f1 - f2) @ (
            table_center_to_player1 - table_center_to_player2
    )
    return inter_f


def vis():
    """ Visualize a HIL or standard trajectory. 
        NOTE: You must have a ground truth demo trajectory to compare against, 
        to see potential deviations in behavior for the given 
        map config (initial pose, goal, and obstacle layout). 

        To elaborate, the workflow might go like this:
        1. Collect a trajectory using the HIL or running robot-robot co-policy.
            During this collection, a ground truth map config is loaded, so the 
            trajectory rollout is given the same map config as the ground truth.
        2. Run this script to visualize the trajectory rollout, and compare it 
            to the ground truth trajectory. Multimodal behavior might occur.
    """
    # ------------------------ Directories Setup ------------------------
    # Setup directories
    if not isdir("results"):
            mkdir("results")
    if not isdir(join("results", "plots")):
        mkdir(join("results", "plots"))

    # ------------------------ Plotting Setup ------------------------   
    # INFO: provide trajectory info
    subject = 2
    ep = 602
    planner_type = "diffusion_policy"
    planner_type2 = "diffusion_policy"
    planner_type3 = "bc_lstm_gmm"
    planner_type4 = "cogail"
    planner_type5 = "vrnn"
    full_path_to_vid_save = "ep-{}-{}".format(ep, planner_type5)

    # INFO: provide info for the traj from the H-H dataset
    og = "datasets/rnd_obstacle_v3/random_run_name_3/ep_602.npz"
    map_cfg = "demo/rnd_obstacle_v3/random_run_name_3/map_cfg/ep_{}.npz".format(ep)
    map_yml = "cooperative_transport/gym_table/config/maps/rnd_obstacle_v3.yml"

    # INFO: provide info for the traj from the HIL data collected starting with f, f2, f3 ...
    f = "results/hil-dp/subject-{}/eval_hil_seed-88_R-planner-{}_H-real-joystick_exp-random_run_name_3_hact-True_subject-{}/eval_hil_seed-88_R-planner-{}_H-real-joystick_exp-random_run_name_3_hact-True_subject-{}-ep_{}.npz".format(subject, planner_type, subject, planner_type, subject, ep)
    f2 = "results/hil-dp/subject-{}/eval_hil_seed-88_R-planner-{}_H-real-joystick_exp-random_run_name_3_hact-False_subject-{}/eval_hil_seed-88_R-planner-{}_H-real-joystick_exp-random_run_name_3_hact-False_subject-{}-ep_{}.npz".format(subject, planner_type2, subject, planner_type2, subject, ep)
    f3 = "results/hil-dp/subject-{}/eval_hil_seed-88_R-planner-{}_H-real-joystick_exp-random_run_name_3_hact-False_subject-{}/eval_hil_seed-88_R-planner-{}_H-real-joystick_exp-random_run_name_3_hact-False_subject-{}-ep_{}.npz".format(subject, planner_type3, subject, planner_type3, subject, ep)
    f4 = "results/hil-dp/subject-{}/eval_hil_seed-88_R-planner-{}_H-real-joystick_exp-random_run_name_3_hact-False_subject-{}/eval_hil_seed-88_R-planner-{}_H-real-joystick_exp-random_run_name_3_hact-False_subject-{}-ep_{}.npz".format(subject, planner_type4, subject, planner_type4, subject, ep)
    f5 = "results/hil-dp/subject-{}/eval_hil_seed-88_R-planner-{}_H-real-joystick_exp-random_run_name_3_hact-False_subject-{}/eval_hil_seed-88_R-planner-{}_H-real-joystick_exp-random_run_name_3_hact-False_subject-{}-ep_{}.npz".format(subject, planner_type5, subject, planner_type5, subject, ep)
    
    skip = 1 # skip every n frames in the trajectory

    max_inter_f = 101 # in data
    min_inter_f = -101

    trajog = dict(np.load(og, allow_pickle=True))["states"][::skip, :2].reshape(-1, 1, 2)
    states = dict(np.load(og, allow_pickle=True))["states"][::skip, :].reshape(-1, 1, 2)
    actions = dict(np.load(og, allow_pickle=True))["actions"][::skip, :2].reshape(-1, 1, 2)
    inter_fog = np.zeros((trajog.shape[0], 1))
    for t in range(trajog.shape[0]):
        inter_f_t = np.abs(compute_interaction_forces(dict(np.load(og, allow_pickle=True))["states"][t], dict(np.load(og, allow_pickle=True))["actions"][t, :2],  dict(np.load(og, allow_pickle=True))["actions"][t, 2:]))
        inter_fog[t] = inter_f_t
    inter_fog = list(inter_fog[::skip][1:-1].flatten())
    inter_fog = np.array([(x - min_inter_f) / (max_inter_f - min_inter_f) * (1 - (-1)) + (-1) for x in inter_fog]) * 101

    traj = dict(np.load(f, allow_pickle=True))["states"][::skip, :2].reshape(-1, 1, 2)
    inter_f = dict(np.load(f, allow_pickle=True))["fluency"][0]["inter_f"][::skip][1:-1]
    inter_f = np.array([(x - min_inter_f) / (max_inter_f - min_inter_f) * (1 - (-1)) + (-1) for x in inter_f]) * 101

    # # load traj2 traj
    # traj2 = dict(np.load(traj2_f, allow_pickle=True))["states"][::skip, :2].reshape(-1, 1, 2)
    # traj2_inter_f = dict(np.load(traj2_f, allow_pickle=True))["fluency"]["inter_f"][::skip][1:-1]
    # traj2_inter_f = np.array([(x - min_inter_f) / (max_inter_f - min_inter_f) for x in inter_f])

    traj2 = dict(np.load(f2, allow_pickle=True))["states"][::skip, :2].reshape(-1, 1, 2)
    inter_f2 = dict(np.load(f2, allow_pickle=True))["fluency"][0]["inter_f"][::skip][1:-1]
    inter_f2 = np.array([(x - min_inter_f) / (max_inter_f - min_inter_f) * (1 - (-1)) + (-1) for x in inter_f2]) * 101

    traj3 = dict(np.load(f3, allow_pickle=True))["states"][::skip, :2].reshape(-1, 1, 2)
    inter_f3 = dict(np.load(f3, allow_pickle=True))["fluency"].item()["inter_f"][::skip][1:-1]
    inter_f3 = np.array([(x - min_inter_f) / (max_inter_f - min_inter_f) * (1 - (-1)) + (-1) for x in inter_f3]) * 101

    traj4 = dict(np.load(f4, allow_pickle=True))["states"][::skip, :2].reshape(-1, 1, 2)
    inter_f4 = dict(np.load(f4, allow_pickle=True))["fluency"].item()["inter_f"][::skip][1:-1]
    inter_f4 = np.array([(x - min_inter_f) / (max_inter_f - min_inter_f) * (1 - (-1)) + (-1) for x in inter_f4]) * 101

    traj5 = dict(np.load(f5, allow_pickle=True))["states"][::skip, :2].reshape(-1, 1, 2)
    inter_f5 = dict(np.load(f5, allow_pickle=True))["fluency"].item()["inter_f"][::skip][1:-1]
    inter_f5 = np.array([(x - min_inter_f) / (max_inter_f - min_inter_f) * (1 - (-1)) + (-1) for x in inter_f5]) * 101

    max_l = max([len(trajog), len(traj), len(traj2), len(traj3), len(traj4), len(traj5)])

    for t in range(5, max_l):
        hspace, vspace = (WINDOW_W / 100, WINDOW_H / 100)
        fig, ax = plt.subplots(figsize=(hspace, vspace), dpi=200)
        plt.rcParams["figure.figsize"] = (hspace, vspace)
        plt.rcParams["axes.edgecolor"] = "black"
        plt.rcParams["axes.linewidth"] = 2.5
        # Create a continuous norm to map from data points to colors
        if traj.shape[0] < t:
            segments = np.concatenate([traj[:-1, :, :2], traj[1:, :, :2]], axis=1)
        else:
            segments = np.concatenate([traj[:t-1, :, :2], traj[1:t, :, :2]], axis=1)
        if traj2.shape[0] < t :
            traj2_segments = np.concatenate([traj2[:-1, :, :2], traj2[1:, :, :2]], axis=1)
        else:
            traj2_segments = np.concatenate([traj2[:t-1, :, :2], traj2[1:t, :, :2]], axis=1)
        if traj3.shape[0] < t :
            traj3_segments = np.concatenate([traj3[:-1, :, :2], traj3[1:, :, :2]], axis=1)
        else:
            traj3_segments = np.concatenate([traj3[:t-1, :, :2], traj3[1:t, :, :2]], axis=1)
        if traj4.shape[0] < t :
            traj4_segments = np.concatenate([traj4[:-1, :, :2], traj4[1:, :, :2]], axis=1)
        else:
            traj4_segments = np.concatenate([traj4[:t-1, :, :2], traj4[1:t, :, :2]], axis=1)
        if traj5.shape[0] < t :
            traj5_segments = np.concatenate([traj5[:-1, :, :2], traj5[1:, :, :2]], axis=1)
        else:
            traj5_segments = np.concatenate([traj5[:t-1, :, :2], traj5[1:t, :, :2]], axis=1)
        if trajog.shape[0] < t :
            trajog_segments = np.concatenate([trajog[:-1, :, :2], trajog[1:, :, :2]], axis=1)
        else:
            trajog_segments = np.concatenate([trajog[:t-1, :, :2], trajog[1:t, :, :2]], axis=1)
        norm = plt.Normalize(min_inter_f, max_inter_f)
        lc = LineCollection(segments, cmap='turbo', norm=norm)
        traj2_lc = LineCollection(traj2_segments, cmap='turbo', norm=norm)
        traj3_lc = LineCollection(traj3_segments, cmap='turbo', norm=norm)
        traj4_lc = LineCollection(traj4_segments, cmap='turbo', norm=norm)
        traj5_lc = LineCollection(traj5_segments, cmap='turbo', norm=norm)
        trajoglc = LineCollection(trajog_segments, cmap='turbo', norm=norm)

        ### INFO: Can plot ALL inter_f segments simultaneously by uncommenting the following lines; otherwise, just uncomment the LineCollection for the method of interest

        # Set the values used for colormapping
        # lc.set_array(inter_f)
        # lc.set_linewidth(2)
        # line = ax.add_collection(lc)

        # traj2_lc.set_array(inter_f2)
        # traj2_lc.set_linewidth(2)
        # line = ax.add_collection(traj2_lc)

        # traj3_lc.set_array(inter_f3)
        # traj3_lc.set_linewidth(2)
        # line = ax.add_collection(traj3_lc)

        # traj4_lc.set_array(inter_f4)
        # traj4_lc.set_linewidth(2)
        # line = ax.add_collection(traj4_lc)

        # traj5_lc.set_array(inter_f5)
        # traj5_lc.set_linewidth(2)
        # line = ax.add_collection(traj5_lc)
        
        trajoglc.set_array(inter_fog)
        trajoglc.set_linewidth(2)
        line = ax.add_collection(trajoglc)
        
        # gt = dict(np.load(gt_f, allow_pickle=True))["states"][::skip]

        # load map info
        map_run = dict(np.load(map_cfg, allow_pickle=True))
        # table initial pose
        table_init = np.zeros(2)
        table_init[0] = map_run["table"].item()["x"]
        table_init[1] = map_run["table"].item()["y"]
        # table goal pose
        table_goal = np.zeros(2)
        table_goal[0] = map_run["goal"].item()["goal"][0]
        table_goal[1] = map_run["goal"].item()["goal"][1]
        # table obstacles as encoding
        num_obs = map_run["obstacles"].item()["num_obstacles"]
        obs = np.zeros((num_obs, 2))
        obstacles = map_run["obstacles"].item()["obstacles"]

        #colorbar
        # divider = make_axes_locatable(ax)
        im_ratio = WINDOW_H / WINDOW_W
        fig.colorbar(line, ax=ax, orientation='vertical', label='Scaled Interaction Forces') #, fraction=0.1*im_ratio)
        
        # ax.annotate('Dec-traj2', xy=(80, 80), xycoords='figure points')
        # ax.annotate('VRNN', xy=(180, 80), xycoords='figure points')
        ax.set_xlim(0, WINDOW_W)
        ax.set_ylim(0, WINDOW_H)
        # plot map
        ca = plt.gca()
        ca.add_patch(
            patches.Circle(
                (traj[0, :, 0], traj[0, :, 1]),
                radius=obstacle_size,
                facecolor=(175 / 255, 175 / 255, 175 / 255, 1.0),  # black
                zorder=0,
            )
        )

        for i in range(obstacles.shape[0]):
            obstacle_w = obstacle_size
            obstacle_h = obstacle_size
            obstacle_x = obstacles[i, 0]  # - obstacle_w / 2.0
            obstacle_y = obstacles[i, 1]  # - obstacle_h / 2.0
            if obstacle_x == 0 or obstacle_y == 0:
                continue
            ca.add_patch(
                patches.Rectangle(
                    (obstacle_x - obstacle_w / 2, obstacle_y + obstacle_h / 2),
                    obstacle_w,
                    obstacle_h,
                    facecolor=(230 / 255, 111 / 255, 81 / 255, 1.0),
                    zorder=0,
                )
            )
        ca.add_patch(
            patches.Rectangle(
                (table_goal[0] - 200 / 2, table_goal[1] - 250 / 2),
                200,
                250,
                facecolor=(242 / 255, 220 / 255, 107 / 255, 1.0),  # gold
                zorder=0,
            )
        )
        plt.gca().set_aspect("equal") 
        plt.axis("off")

        plot_dir = join(
                "results",
                "plots",
                "ep_{}-inter_f".format(ep)
            )
        if not isdir(plot_dir):
            mkdir(plot_dir)
        
        plot_name = join(
                "results",
                "plots",
                "ep_" + str(ep) + "-inter_f",
                "{0}".format(t),
            )



        plt.xlabel("xlabel", fontsize=18)
        plt.ylabel("ylabel", fontsize=16)
        fig.set_size_inches(10, 5)
        plt.savefig(plot_name, dpi=200)
        # plt.show()
        plt.close('all')


    # ------------------------ Video --------------------------------

    make_video(plot_dir, "ep_{}-inter_f".format(ep))
        


def main():
    vis()


if __name__ == "__main__":
    main()