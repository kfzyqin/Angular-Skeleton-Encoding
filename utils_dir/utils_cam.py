import torch
import matplotlib.pyplot as plt
import seaborn as sns


def plot_freq(cam_dict):
    total_num = 0
    for a_key in cam_dict:
        a_action_cam = cam_dict[a_key].cpu().numpy()
        print('action: ', a_key, 'cam shape: ', a_action_cam.shape)
        total_num += a_action_cam.shape[0]
        ax = sns.heatmap(a_action_cam)
        plt
        plt.savefig(f'test_fields/action_heatmaps/action_{a_key}_heatmap.png', dpi=200,
                    bbox_inches='tight')
        # plt.show()
        plt.close()
    print('total num: ', total_num)
