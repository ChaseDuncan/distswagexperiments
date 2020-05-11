import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import softmax

def parse(npz_arr):
    d = np.load(npz_arr)
    return d['predictions'], d['targets']

def calibration_curve(npz_arr, num_bins):
    outputs, labels = parse(npz_arr)
    if outputs is None:
        out = None
    else:
        confidences = np.max(outputs, 1)
        step = (confidences.shape[0] + num_bins - 1) // num_bins
        bins = np.sort(confidences)[::step]
        if confidences.shape[0] % step != 1:
            bins = np.concatenate((bins, [np.max(confidences)]))
        predictions = np.argmax(outputs, 1)
        bin_lowers = bins[:-1]
        bin_uppers = bins[1:]

        accuracies = predictions == labels

        xs = []
        ys = []
        zs = []

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = (confidences > bin_lower) * (confidences < bin_upper)
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                xs.append(avg_confidence_in_bin)
                ys.append(accuracy_in_bin)
                zs.append(prop_in_bin)
        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)

        out = {"confidence": xs, "accuracy": ys, "p": zs, "ece": ece}
    return out

if __name__=='__main__':
    #import os
    #for root, dirs, files in os.walk("experiments/", topdown=False):
    #    for name in files:
    #        if ".npz" in name:
    #            print(os.path.join(root, name))
    #            out = calibration_curve(os.path.join(root, name), 20)
    #            conf_acc_diff = out['confidence'] - out['accuracy']
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=argparse.FileType('r'), nargs='+')
    parser.add_argument('--num_bins', type=int, default=20)
    args = parser.parse_args()
    print(args.file)

    fig, ax = plt.subplots()
    for f in args.file:
        out = calibration_curve(f.name, args.num_bins)
        conf_acc_diff = out['confidence'] - out['accuracy']
        #print(out['confidence'])
        #print(f'confidence: {out["confidence"]}')
        #print(f'conf-accry: {conf_acc_diff}')

        # Data for plotting
        ax.plot(out['confidence'], conf_acc_diff, marker='^')
        break

    ax.grid()
    ax.set(xlabel='Confidence', ylabel='Confidence - Accuracy',
              title='Calibration')
    fig.savefig("test.png")
    plt.show()
