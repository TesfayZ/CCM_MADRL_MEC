The three folders correspond to the three figures on the paper. 
The "s10lre4e3CCM_MADRL_MEC" stands for 10 steps per episode, and learning rages of 0.0001 and 0.001 for the cllient and master agents respectively. 
In side the folder, the CSV folder stores the the performance of the experiment at each episode. 
The results are stored for both the evaluation environment and the trainig environment but only the evaluation environmenet is discussed in the paper.  
The "Figures" folder stores the ouput of the plotting scripts. 
The plotAtTraning plots the result using the traning environment whereas the other scripts plot the result using the evaluation environment. 


The CCM_MADRL_MEC can be run as <python run.py index> where index=0 stores the result to a CCM_MADRL_MEC0.csv file. 
To run multiple experiments, the index values must must start from 0 and increament untl the number of runs so that they are easily accessed for the plotting scripts.

The benchmarks can run one by one using <python Benchmarks_run.py index> by replacing the Benchmarks_modes in line 37 of Benchmarks_run.py with one of the choices after the comment. 

To plot the result, adjust the numebr of runs in line 93 of the plot3.py and run as <python plot3.py> 
