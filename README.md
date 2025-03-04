# Ipsae analysis of competition designs 

This repo holds the code to parse the AlphaPulldown results, calculate ipsae, and make figures for the correlation and ROCAUC curve. Ipsae code is taken from the [original repository](https://github.com/DunbrackLab/IPSAE) but crudely modified to return the max ipsae value rather than log the results. 

## Requirements
- Jupyter Notebook
- NumPy, Pandas, Matplotlib, Scikit-learn

## Notes

The unzipped AlphaPulldown results should be placed in the ```/data``` directory.

Run the ```calculate_ipsae.ipynb``` notebook to add the ipsae scores as a new column to the results file released by AdaptyvBio (```/data/result_summary.csv```)'

Then, run the ```plot_curves.ipynb``` notebook to make the figures for the correlation between ipsae and kd, as well as the ROCAUC curve for binding. 
