# Electricity Price Forecasting on the Day-Ahead market using Machine Learning

This repository contains the code necessary to reproduce the experiments of the paper. Please cite us using. 

## Installation

Create a virtul env using python:
> python -m venv .

Install all dependencies:
> python -m pip install -r requirements.txt

This code uses the EPFToolbox that you need to install from [this](https://github.com/jeslago/epftoolbox) repository.

Start by setting a env variable for storing results and load data in a python iterpreter:
> import os
> os.environ["VOLTAIRE"] = os.curdir

The Scripts will populate the 'data/dataset' folder with results and retrieved datasets

## Data download

You can download the data from a [DropBox archive](https://www.dropbox.com/sh/2n7qje9dmhixh35/AADffdnjmJXRQEdvxbcBECgma?dl=0). Extract it in this repository's folder.

## Scripts

'LAGO_RESULTS.py' will reproduce metrics for the epftoolbox issued-models.
'TSCHORA_RESULTS.py' will reproduce metrics for our version of the ML models.
