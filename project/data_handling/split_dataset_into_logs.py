import argparse
import os
import platform
import pandas as pd


machine = platform.system()
path_to_windows = lambda path: path.replace('/', '\\')
parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default="/home/linuxpc/MEGAsync/all_data_tesi/data")


if __name__ == "__main__":
    args = parser.parse_args()
    base_dir = args.base_dir if machine == "Linux" else path_to_windows(args.base_dir) 
    
    csv = pd.read_csv(os.path.join(base_dir, "data.csv"))
    logs = csv["log"].unique()

    csvs = os.path.join(base_dir, "..", "for_tgn")
    if not os.path.exists(csvs):
        os.mkdir(csvs)

    pad = len(str(len(logs)))
    for log in logs:
        df = csv[csv["log"]==log].copy()
        df = df.drop(["case", "log"], axis=1)
        df.to_csv(os.path.join(csvs, f"log_{str(log).zfill(pad)}.csv"))