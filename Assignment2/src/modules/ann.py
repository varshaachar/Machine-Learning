import sys
sys.path.append('./ABAGAIL/ABAGAIL.jar')
from shared import Instance
import csv
import time


def initialize_instances(infile):
    """Read the m_trg.csv CSV data into a list of instances."""
    instances = []

    # Read in the CSV file
    with open(infile, "r") as dat:
        reader = csv.reader(dat)

        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            instance.setLabel(Instance(0 if float(row[-1]) <= 0 else 1))
            instances.append(instance)

    return instances

def train(oa, network, oaName, training_ints, testing_ints, measure, TRAINING_ITERATIONS, OUTFILE):
    """Train a given network on a set of instances.
    """
    print "\nError results for %s\n---------------------------" % (oaName,)
    times = [0]
    for iteration in xrange(TRAINING_ITERATIONS):
        start = time.clock()
        oa.train()
        elapsed = time.clock()-start
    	times.append(times[-1]+elapsed)
        if iteration % 10 == 0:
    	    MSE_trg, acc_trg, F1_trg = errorOnDataSet(network,training_ints,measure)
            MSE_tst, acc_tst, F1_tst = errorOnDataSet(network,testing_ints,measure)
            txt = '{},{},{},{},{},{},{},{}\n'.format(iteration,MSE_trg,MSE_tst,acc_trg,acc_tst,F1_trg,F1_tst,times[-1])
            with open(OUTFILE,'a+') as f:
                f.write(txt)

def errorOnDataSet(network,ds,measure):
    N = len(ds)
    error = 0.
    correct = 0
    incorrect = 0
    false_positives = 0.0
    false_negatives = 0.0
    true_positives = 0.0
    true_negatives = 0.0
    for instance in ds:
        network.setInputValues(instance.getData())
        network.run()
        actual = instance.getLabel().getContinuous()
        predicted = network.getOutputValues().get(0)
        predicted = max(min(predicted,1),0)

        # Measure type of error for F1 score
        if actual == 0.0 and predicted >= 0.5:
            false_positives += 1.0

        if actual == 0.0 and predicted < 0.5:
            true_negatives += 1.0

        if actual == 1.0 and predicted >= 0.5:
            true_positives += 1.0

        if actual == 1.0 and predicted < 0.5:
            false_negatives += 1.0


        if abs(predicted - actual) < 0.5:
            correct += 1
        else:
            incorrect += 1
        output = instance.getLabel()
        output_values = network.getOutputValues()
        example = Instance(output_values, Instance(output_values.get(0)))
        error += measure.value(output, example)
    MSE = error/float(N)
    acc = correct/float(correct+incorrect)

    try:
        precision = true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        precision = 0.0

    try:
        recall = true_positives / (true_positives + false_negatives)
    except ZeroDivisionError:
        recall = 0.0

    try:
        F1 = 2.0 * ((precision * recall) / (precision + recall))
    except ZeroDivisionError:
        F1 = 0.0

    return MSE,acc,F1
