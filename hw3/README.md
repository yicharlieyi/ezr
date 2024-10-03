# HW3

## Setting Up

I'm on windows and using code space.


### Installation
Fork the code:

    repo url: https://github.com/yicharlieyi/ezr

Open code space and cd into the /workspaces/ezr/hw3 directory

Make sure python 3.13 and pytest is installed, to do that, type in the command line:

    sudo apt update -y; sudo  apt upgrade -y; sudo apt install software-properties-common -y; sudo add-apt-repository ppa:deadsnakes/ppa -y ; sudo apt update -y ; sudo apt install python3.13 -y
and 

    pip install pytest

###  Running the code 

Run the experiment:
    
For example, run it on a small dimensional data set:
        
    python3.13 -B experiment.py -t /workspaces/ezr/data/optimize/config/SS-H.csv

In order to run it on another data set, **update the file path in line 13 of the experiment.py** file and the command line

    python3.13 -B experiment.py -t /workspaces/ezr/data/optimize/config/SS-N.csv
    
To run the test cases:

    pytest test_experiment.py 


### The rq.sh Results and Conclusion
<img width="817" alt="Screen Shot 2024-09-19 at 12 33 52 AM" src="https://github.com/user-attachments/assets/86b95bdb-520d-4ff3-863c-a95ebafd6db6">

These are the results from the rq.sh results where it contains rows such as RANKS where it tells you how often treatments are in rank 0,1,2,etc, EVALS which is the budgets used to achieve those ranks, and DELTAS which is just the formula, 100*(asIs - now)/asIs change, to get the results. The results presented in this table gives us an asIs output where it represents the baseline performance without any modifications to the data or method. These results shows us that exploitation and exploration strategies with b=True (branching) are generally more effective than random guessing which random guessing was the 2nd hypothesis. Since we observed JJR1, we confirm their hypothesis.
