import json, pickle

dict = {}

def json_to_pkl(json_path: str):
    with open(json_path, "rb") as f:
        json_data = json.load(f)
    with open(file=json_path.replace("json", "pkl"), mode='wb') as f:
        pickle.dump(json_data, f)
    

if __name__ == "__main__":
    json_path = "data/smpl walk zpos edited 0.5%.json"
    json_to_pkl(json_path=json_path)
