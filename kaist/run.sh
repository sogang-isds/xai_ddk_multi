#!/bin/bash

# save all as dataframe
python -m kaist.save_labels
python -m kaist.save_data
python -m kaist.save_preds

# shap
python -m kaist.save_shap
python -m kaist.evaluate_shap

# ig
python -m kaist.save_ig
python -m kaist.evaluate_ig