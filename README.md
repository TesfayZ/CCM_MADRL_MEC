The code for the paper [Combinatorial Client-Master Multiagent Deep Reinforcement Learning for Task Offloading in Mobile Edge Computing](https://arxiv.org/abs/2402.11653). Detailed description is available in the 5th chapter of my [PhD thesis](https://scholar.google.gr/citations?view_op=view_citation&hl=en&user=17hsRTkAAAAJ&citation_for_view=17hsRTkAAAAJ:UeHWp8X0CEIC).

The three folders correspond to the three figures on the paper. 
The "s10lre4e3CCM_MADRL_MEC" stands for 10 steps per episode, and learning rates of 0.0001 and 0.001 for the cllient and master agents respectively. 
Inside the folder, the CSV folder stores the performance of the experiment every episode. 
The results are stored for both the evaluation environment and the trainig environment but only the evaluation environmenet is discussed in the paper.  
The "Figures" folder stores the output of the plotting scripts. 
The plotAtTraning.py plots the result using the traning environment whereas the other scripts plot the result using the evaluation environment. 


Run the CCM_MADRL_MEC as python run.py index where index=0 stores the result to a CCM_MADRL_MEC0.csv file. 
To run multiple experiments, the index values must must start from 0 and increament until the number of runs so that they are accessed sequentiality for the plotting scripts.

Run the benchmark algorithms one by one using python Benchmarks_run.py index by replacing the Benchmarks_modes in line 37 of Benchmarks_run.py with one of the choices in the commented text. 

To plot the result, adjust the number of runs in line 93 of the plot3.py and run as python plot3.py

Note: For the 100 steps per episode experiments ("s100lre4e3CCM_MADRL_MEC", and "s100lre4e3CCM_MADRL_MECLabmbda5"), use the s100lre4e3CCM_MADRL_MECLabmbda5 for your analysis of the experiment. The one with lambda1=Lambda2=0.5 has neglected the energy comsumption becuase the value is small compared to the latency, and hence only latency affects the performance. lambda1=1 and Lambda2=5 provides better balance between the two costs. More explanations will come in the future version. 
