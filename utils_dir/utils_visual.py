import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch

azure_kinect_bone_pairs = (
    (1, 0), (2, 1), (3, 2), (4, 2), (5, 4), (6, 5), (7, 6), (8, 7), (9, 8), (10, 7), (11, 2), (12, 11),
    (13, 12), (14, 13), (15, 14), (16, 15), (17, 14), (18, 0), (19, 18), (20, 19), (21, 20), (22, 0),
    (23, 22), (24, 23), (25, 24), (26, 3), (27, 26), (28, 27), (29, 28), (30, 27), (31, 30)
)

kinect_v2_bone_pairs = tuple((i - 1, j - 1) for (i, j) in (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
))

bone_pair_dict = {
    'azure_kinect': azure_kinect_bone_pairs,
    'kinect_v2': kinect_v2_bone_pairs
}


def azure_kinect_post_visualize(frames, save_name=None, sklt_type='azure_kinect'):
    min_x, max_x = torch.min(frames[0][0]).item(), torch.max(frames[0][0]).item()
    min_y, max_y = torch.min(frames[0][1]).item(), torch.max(frames[0][1]).item()
    min_z, max_z = torch.min(frames[0][2]).item(), torch.max(frames[0][2]).item()

    bones = bone_pair_dict[sklt_type]
    def animate(skeletons):
        # Skeleton shape is 3*25. 3 corresponds to the 3D coordinates. 25 is the number of joints.
        ax.clear()

        # ax.set_xlim([-3000, 1000])
        # ax.set_ylim([-3000, 1000])
        # ax.set_zlim([-3000, 1000])

        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])
        ax.set_zlim([min_z, max_z])

        # ax.set_xlim([-2, -1])
        # ax.set_ylim([2, 3])
        # ax.set_zlim([-2, 0])

        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])

        # person 1
        k = 0
        color_list = ('blue', 'orange', 'cyan', 'purple')
        # color_list = ('blue', 'blue', 'cyan', 'purple')
        color_idx = 0

        while k < skeletons.shape[0]:
            for i, j in bones:
                joint_locs = skeletons[:, [i, j]]
                # plot them
                ax.plot(joint_locs[k+0], joint_locs[k+1], joint_locs[k+2], color=color_list[color_idx])
                # ax.plot(-joint_locs[k+0], -joint_locs[k+2], -joint_locs[k+1], color=color_list[color_idx])

            k += 3
            color_idx = (color_idx + 1) % len(color_list)

        # Rotate
        # X, Y, Z = axes3d.get_test_data(0.1)
        # ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)
        #
        # # rotate the axes and update
        # for angle in range(0, 360):
        #     ax.view_init(30, angle)

        if save_name is None:
            title = 'Action Visualization'
        else:
            title = os.path.split(save_name)[-1]
        plt.title(title)
        skeleton_index[0] += 1
        return ax

    for an_entry in range(1):
        if isinstance(an_entry, tuple) and len(an_entry) == 2:
            index = int(an_entry[0])
            pred_idx = int(an_entry[1])
        else:
            index = an_entry
        # get data
        skeletons = np.copy(frames[index])

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # ax.set_xlim([-1, 1])
        # ax.set_ylim([-1, 1])
        # ax.set_zlim([-1, 1])

        # print(f'Sample index: {index}\nAction: {action_class}-{action_name}\n')  # (C,T,V,M)

        # Pick the first body to visualize
        skeleton1 = skeletons[..., 0]  # out (C,T,V)
        # make it shorter
        shorter_frame_start = 0
        shorter_frame_end = 300
        if skeletons.shape[-1] > 1:
            skeleton2 = np.copy(skeletons[..., 1])  # out (C,T,V)
            # make it shorter
            skeleton2 = skeleton2[:, shorter_frame_start:shorter_frame_end, :]
            # print('max of skeleton 2: ', np.max(skeleton2))
            skeleton_frames_2 = skeleton2.transpose(1, 0, 2)
        else:
            skeleton_frames_2 = None

        skeleton_index = [0]
        skeleton_frames_1 = skeleton1[:, shorter_frame_start:shorter_frame_end, :].transpose(1, 0, 2)

        if skeleton_frames_2 is None:
            # skeleton_frames_1 = center_normalize_skeleton(skeleton_frames_1)
            ani = FuncAnimation(fig, animate,
                                skeleton_frames_1,
                                interval=150)
        else:
            # skeleton_frames_1 = center_normalize_skeleton(skeleton_frames_1)
            # skeleton_frames_2 = center_normalize_skeleton(skeleton_frames_2)
            ani = FuncAnimation(fig, animate,
                            np.concatenate((skeleton_frames_1, skeleton_frames_2), axis=1),
                            interval=150)

        if save_name is None:
            save_name = 'tmp_skeleton_video_2.mp4'
        print('save name: ', save_name)
        ani.save(save_name, dpi=200, writer='ffmpeg')
        plt.close('all')


def plot_multiple_lines(lines, save_name=None, labels=None, every_n=1):
    font = {'size': 30}
    import matplotlib
    matplotlib.rc('font', **font)

    if labels is not None:
        assert len(lines) == len(labels)
    markers = ['^', '.', '*']
    # colors = ['#dd0100', '#225095', '#fac901']
    colors = ['#dd0100']

    # plt.xlim(0, len(lines[0]))  # Chronological loss value
    # plt.ylim(-0.25, 1.25)  # Chronological loss value

    plt.xlim(0, len(lines[0]))
    plt.ylim(0, 9)

    x_axis_list = list([x for x in range(0, len(lines[0]), every_n)])
    for line_idx, a_line in enumerate(lines):
        a_line_plot = list([a_line[i] for i in range(0, len(lines[0]), every_n)])
        plt.plot(x_axis_list, a_line_plot,
                 color=colors[line_idx % len(colors)],
                 marker=markers[line_idx % len(markers)],
                 markersize=30,
                 label=labels[line_idx] if labels is not None else None)

    # plt.xticks(x_axis_list, a_line_plot)
    plt.grid()
    # plt.legend(loc=4)  # Chronological loss value

    fig = matplotlib.pyplot.gcf()

    # fig.set_size_inches(7, 10)  # Chronological loss value
    fig.set_size_inches(10, 8)

    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name, bbox_inches='tight')
        plt.close()
        plt.show()
