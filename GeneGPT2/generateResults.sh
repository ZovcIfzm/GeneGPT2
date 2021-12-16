#!/bin/bash

python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
cd GeneGPT2

python3 trainGPT2.py
python3 XGBoostRegressor.py --dataset_type raw_c > ../raw_c_results.txt
python3 XGBoostRegressor.py --dataset_type raw_d > ../raw_d_results.txt
python3 XGBoostRegressor.py --dataset_type raw > ../raw_results.txt