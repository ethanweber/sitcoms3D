# Convert SMPL pkl file to be compatible with Python 3
# Script is from https://rebeccabilbro.github.io/convert-py2-pickles-to-py3/
import os
import shutil
import subprocess
import dill
import pickle


def convert(old_pkl):
    """
    Convert a Python 2 pickle to Python 3
    """
    # Make a name for the new pickle
    new_pkl = os.path.splitext(os.path.basename(old_pkl))[0] + "_p3.pkl"

    # Convert Python 2 "ObjectType" to Python 3 object
    dill._dill._reverse_typemap["ObjectType"] = object

    # Open the pickle using latin1 encoding
    with open(old_pkl, "rb") as f:
        loaded = pickle.load(f, encoding="latin1")

    # Re-save as Python 3 pickle
    with open(new_pkl, "wb") as outfile:
        pickle.dump(loaded, outfile)


_ = subprocess.call(
    "wget https://github.com/classner/up/raw/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl", shell=True)
convert('basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
_ = subprocess.call("ls")
os.remove("basicModel_neutral_lbs_10_207_0_v1.0.0.pkl")
os.makedirs("data/smpl_models/smpl", exist_ok=True)
shutil.move("basicModel_neutral_lbs_10_207_0_v1.0.0_p3.pkl", "data/smpl_models/smpl/SMPL_NEUTRAL.pkl")
