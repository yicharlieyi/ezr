# Easier  AI (just the important bits)


&copy; 2024 Tim Menzies, timm@ieee.org     
BSD-2 license. Share and enjoy.  

----------------------------------

For over two decades, I have been mentoring people about SE and AI.
When you do that, after a while, you realize:

- When it is all said and done, you only need  a dozen or so cool tricks;
- Other people really only need a  few dozen or so bits of AI theory;
- Everyone  could have more fun, and get more done, if we avoided
  the same dozen or so traps.

So I decided to write down that theory and those tricks and    traps
(see below).  I took some XAI code (explainable AI) I'd written for
semi-supervised multiple-objective optimization. Then I wrote notes
on any part of the code where  I had spent time helping helping
people with  those tricks, theory and traps.

Here is how the notes are labelled. For way-out ideas, read the 500+ ones.
For good-old-fashioned command-line warrior stuff, see 100-200

- Odd number items are about SE;
- So even numbers are about AI;
- 

|Anit-patterns<br>(things not to do) | SE system | SE coding | AI coding | AI theory<br>(standard) | New AI ideas| 
|:----------------------------------:|:---------:|:---------:|:---------:|:-----------------------:|:-----------:|
|00 - 99                             | 100 - 199 |  200-299  | 300-399   | 400 - 499               |  500-599    | 


One more thing.  The SE and AI literature is full of bold experiments
that try a range of new ideas.  But some new ideas are better than
others. With all little time, and lots of implementation experience,
we can focus of which  ideas offer the "most bang per buck".

Share and enjoy.
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
