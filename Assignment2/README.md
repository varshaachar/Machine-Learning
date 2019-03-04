# HW2 - Jython Implementation

To adapt the `ANN_*.py` files to your dataset:
- Change the `INPUT_LAYER` variable to reflect the number of features.
- Replace `bank_test.csv` and `bank_train.csv` in each file with your dataset.

Run with

```
jython ANN_back.py
```

- Add support for parameters if you got too much time
- Push helpful changes to this repo

---------------


Please add anything you find useful for coding up hw2 using scikit-learn or Python implementation thank you.


https://github.com/skylergrammer/SimulatedAnnealing -- scikit learn simulated annealing
https://github.com/rsteca/sklearn-deap -- scikit learn genetic algorithm

MIMIC implementation in Python:
https://github.com/mjs2600/mimicry

Isbell's Paper on MIMIC:
https://www.cc.gatech.edu/~isbell/papers/isbell-mimic-nips-1997.pdf


---------------------------------------------------
Winnie's Approach to Installing Dependencies on Windows
1) Install Java SDK
https://www.oracle.com/technetwork/java/javase/downloads/jdk11-downloads-5066655.html
2) Install Apache Ant
https://ant.apache.org/bindownload.cgi

- locate folders path of Java and Ant Apache 
- Add JAVA_HOME and ANT_HOME and append to PATH in environment variables
(Follow this instruction: https://www.mkyong.com/ant/how-to-install-apache-ant-on-windows/)

3) Install Jython
https://www.jython.org/archive/22/installation.html
- locate folder path of Jython
- Add JYTHON_HOME and append to PATH
(Follow hygull's screenshot in stack overflow: https://stackoverflow.com/questions/8148780/how-do-i-set-the-environment-variables-for-jython)
- Test in any terminal to see if jython can be called upon

4) Git clone Abigail
- Open terminal window to the cloned folder where build.xml is located
- Run ant to compile the .jar file

5) Git clone this repository
- Copy the entire Abigail folder into src
- Each file can be run calling "jython ANN_xxxxxxx.py"

 
