"""
Useless for now, might need later when training and test data are separate .root files
"""

import pandas as pd
import uproot

tree_dir = "/hepgpu4-data1/yuliia/"
tree_file = tree_dir + "spg_photon.csv"
tree_csv = pd.read_csv(tree_file)

input_dir = "/hepgpu4-data1/yuliia/training_output/"
input_file = input_dir + "MPID_scores_gamma_true_vertex_6625_steps.csv"
input_csv = pd.read_csv(input_file)

input_csv['run'] = tree_csv['run']
input_csv['subrun'] = tree_csv['subrun']
input_csv['event'] = tree_csv['event']
input_csv['Energy'] = tree_csv['Energy']
input_csv['Momentum'] = tree_csv['Momentum']
input_csv['V_x'] = tree_csv['V_x']
input_csv['V_y'] = tree_csv['V_y']
input_csv['V_z'] = tree_csv['V_z']

output_dir = input_dir
output_file = output_dir + "MPID_scores_gamma_additional_6625_steps.csv"

input_csv.to_csv(output_file, index=False)
