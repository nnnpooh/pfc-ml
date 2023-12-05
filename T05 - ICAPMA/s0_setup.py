import os, sys

cwd = sys.path[0]

folders = [
    "o1_pfc_test",
    "o1_pfc_train",
    "o2_train_data",
    "o3_checkpoints",
    "o3_model_params",
    "o4_analyze",
    "o5_visualize",
]

for folder in folders:
    folderPath = os.path.join(cwd, folder)
    if not (os.path.exists(folderPath)):
        os.mkdir(folderPath)
    else:
        print(folderPath, "already exists.")
