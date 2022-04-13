#!/bin/bash

echo "Running CEVAE"
python main.py configs/cevae.json ate

echo "Running DFPV"
python main.py configs/dfpv.json ate

echo "Running KPV"
python main.py configs/kpv.json ate

echo "Running naive linear regression Y ~ A + W + Z"
python main.py configs/linear_regression_AWZY.json ate

echo "Running naive linear regression Y ~ A"
python main.py configs/linear_regression_AY.json ate

echo "Running naive neural net Y ~ A"
python main.py configs/naive_nn_AY_demand.json ate

echo "Running naive neural net Y ~ A + W + Z"
python main.py configs/naive_nn_AWZY_demand.json ate

echo "Running NMMR"
python main.py configs/NMMR.json ate

echo "Running PMMR"
python main.py configs/pmmr.json ate