# RID

## Reinforced Dynamics of solvated systems using DFTB  

Dev Notes: 

TO-BE-DONE: fix plots for monitoring 1) CVs and 2) FES
10-02-2022: if mean force higher than sel_threshold (160) not found after biasMD reduce sel_threshold by 40 KJ till no exception is shown
09-30-2022: parallelized the resMD job submissions. job name is hard coded into run_mp.py