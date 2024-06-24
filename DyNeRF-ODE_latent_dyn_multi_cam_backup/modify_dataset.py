import os
import shutil
import json

"""dataset_path_target = '/hdd5/hiran/NeRF_ODE/DyNeRF-ODE_latent_dyn_multi_cam/DyNeRF_blender_data_ball_roll_multicam/train/'
dataset_path_source = '/hdd5/hiran/NeRF_ODE/DyNeRF-ODE_latent_dyn_multi_cam/DyNeRF_blender_data_ball_roll_multicam/all/'

for k, i in enumerate(os.listdir(dataset_path_source)):
    for j in os.listdir(dataset_path_source + str(i) + '/train/'):
        shutil.copy(dataset_path_source + str(i) + '/' + 'train/' + 'r_' + str(int(j[2:].split('.')[0])) + '.png', dataset_path_target + 'r_' + str(80*k+int(j[2:].split('.')[0])) + '.png')
        
"""
dataset_json_target = '/hdd5/hiran/NeRF_ODE/DyNeRF-ODE_latent_dyn_multi_cam/DyNeRF_blender_data_ball_roll_multicam/transforms_train.json'
dataset_json_source = '/hdd5/hiran/NeRF_ODE/DyNeRF-ODE_latent_dyn_multi_cam/DyNeRF_blender_data_ball_roll_multicam/all/'

new_transform = {"camera_angle_x": 0.6911112070083618, "frames":[]}

for k, i in enumerate(os.listdir(dataset_json_source)):
    with open(dataset_json_source + str(i) + '/transforms_train.json', 'r') as f:
        a = json.load(f)
    for j in a['frames']:
        j['file_path'] = './train/r_' + str(80*k + int(j['file_path'][10:]))
        new_transform["frames"].append(j)

with open(dataset_json_target, 'w') as json_file:
    json.dump(new_transform, json_file)

