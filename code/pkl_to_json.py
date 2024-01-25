import pickle
import json
import numpy as np

def compute_qvel(qpos):
    idx = 0
    qvel = np.empty_like(qpos)
    for idx in range(qpos.shape[0]-1):
        qvel[idx] = (qpos[idx+1] - qpos[idx]) / 0.0083
    qvel[idx+1] = qvel[idx]
    return qvel

def pkl_to_json(pkl_path: str):
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    for key, value in pkl_data.items():
        if type(value) == np.ndarray:
            pkl_data[key] = value.tolist()
    # joint_idxs = [x - 7 for x in [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,39,40,41,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]]
    # pkl_data['qpos'] = np.concatenate((np.array(pkl_data['root_translation']), np.array(pkl_data['rotation'])[:,0,[3,0,1,2]], np.array(pkl_data['qpos'])[:,joint_idxs]), axis=1)
    pkl_data['qpos'] = np.array(pkl_data['qpos'])
    pkl_data['xpos'] = np.array(pkl_data['xpos'])
    pkl_data['xpos'] = pkl_data['xpos'][:, 1:, :]
    pkl_data['qpos'][:, 2] = np.array(pkl_data['root'])[:, 2]
    pkl_data['qvel'] = compute_qvel(pkl_data['qpos'])
    pkl_data['xpos'] = pkl_data['xpos'].tolist()
    pkl_data['qpos'] = pkl_data['qpos'].tolist()
    pkl_data['qvel'] = pkl_data['qvel'].tolist()
    
    # del pkl_data['root_translation']
    # del pkl_data['rotation']
    del pkl_data['root']
    pkl_data['length'] = len(pkl_data['qpos'])
    with open(pkl_path.replace("pkl", "json"), 'w', encoding='utf-8') as make_file:
        json.dump(pkl_data, make_file, indent=4)


if __name__ == "__main__":
    pkl_path = "data/smplrig_cmu_walk_16_15_zpos_edited.pkl"
    pkl_to_json(pkl_path=pkl_path)