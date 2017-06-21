# Python_wh40K_Tau


This is small project I started with mixed goal of:

 * putting my basic python skills to test
 * compare different weapons+system loadouts availalbe to Commander and Cryssis Suits.
 
*.csv files contains hard coded values for commander and crysis (the bits important for this project), for specific weapons+systemts combinations, for specific enemies and different conditions (conditions are not coded in yet)
 
Project runs 10 000 simulation of shooting step for each combination of shooter/loadout/enemy/condition and stores results in a one big table.

At this point only values for number of wounds are stored but simulation also generates values for number of models killed by shooting + number of extra models lost by morale.

After generating the data I try to plot the result using python implementtaion of R's ggplot.
