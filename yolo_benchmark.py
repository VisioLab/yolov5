from vlabml.training.detector.load_data import load_data
from pathlib import Path
import subprocess
import os
import sys
import shutil

def run_yolo_benchmark():
    days = [['hannover_2022-07-18_01_onboarding-instances', 'hannover_2022-07-18_01_checkout-instances'],
            ['hannover_2022-09-06_01_onboarding-instances', 'hannover_2022-09-06_01_checkout-instances']]

    for day in days:
        load_data(data_dir=Path('custom_data/'), display_name=day[0], 
                    display_name_val=day[1], split_size=1.0)
        os.system('python train.py --freeze 10 --img 640 --batch 1 --epochs 1 --data ./custom_data/dataset.yaml --weights ./custom_data/yolo_one_stage_best.pt --cache')
        try:
            shutil.rmtree('./mlruns')
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
        #!python train.py --freeze 10 --img 640 --batch 1 --epochs 1 --data ./custom_data/dataset.yaml --weights ./custom_data/yolo_one_stage_best.pt

if __name__ == "__main__":
    run_yolo_benchmark()