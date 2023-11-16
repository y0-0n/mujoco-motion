import pickle
import json
import numpy as np

def pkl_to_json(pkl_path: str):
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    for key, value in pkl_data.items():
        if type(value) == np.ndarray:
            pkl_data[key] = value.tolist()
    with open(pkl_path.replace("pkl", "json"), 'w', encoding='utf-8') as make_file:
        json.dump(pkl_data, make_file, indent=4)


if __name__ == "__main__":
    pkl_path = "data/M02F4V1.pkl"
    pkl_to_json(pkl_path=pkl_path)