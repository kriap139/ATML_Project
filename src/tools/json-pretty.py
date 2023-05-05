import json
import sys
import os


def prettify(file_path: str):
    with open(file_path, mode='r') as f:
        s = json.load(f)

    directory = os.path.dirname(file_path)
    fn, ext = os.path.splitext(os.path.basename(file_path)) 


    with open(f"{directory}/{fn}_pretty{ext}", mode='w') as f:
        json.dump(s, f, indent=3)
    

if __name__ == "__main__":
    prettify(sys.argv[1])