import numpy as np 
import io 
import pyedflib 
import os 
import glob 
import matplotlib.pyplot as plt 


basedir = "E:/expt/EEG_classify/cnn-eeg/eegmmidb/files"
list_of_files = os.listdir(basedir)
edf_files = glob.glob("%s/%s/*.edf"%(basedir,list_of_files[1]))
path_edf = os.path.join(basedir,list_of_files[1],edf_files[10])

try:
    sig = pyedflib.EdfReader(path_edf)
    print(path_edf)
    n = sig.signals_in_file 
    print(n)
    signal_labels = sig.getSignalLabels()
    print("signal labels",signal_labels)
    sigbuf = np.zeros((n,sig.getNSamples()[0]))
    print("sigbuf shape",sigbuf.shape)
    for j in np.arange(n):
        sigbuf[j,:]=sig.readSignal(j)

    print("signal",sigbuf)
    #print("1 signal shape", sigbuf[0].shape)
    annotations = sig.read_annotation()
    print("annotation",annotations)
except KeyboardInterrupt:
    sig._close()
    raise 
sig._close()
del sig

sigbuf = sigbuf.transpose()

print(len(sigbuf[... ,1]))  #data of one row

#plt.plot(sigbuf[... ,1])   
#plt.show()


######################################################
########################################################

num_classes=2
long_edge=False

SAMPLE_RATE = 160 
EEG_CHANNELS = 64

BASELINE_RUN = 1
MI_RUNS = [4, 8, 12] # l/r fist
if num_classes >= 4:
    MI_RUNS += [6, 10, 14] # feet (& fists)
    
# total number of samples per long run
RUN_LENGTH = 125 * SAMPLE_RATE 
# length of single trial in seconds
TRIAL_LENGTH = 6 if not long_edge else 10
NUM_TRIALS = 21 * num_classes 

n_runs = len(MI_RUNS)
X = np.zeros((n_runs, RUN_LENGTH, EEG_CHANNELS))
events = []

for i_run, current_run in enumerate(MI_RUNS):
    signals = sigbuf  #needed return value of load_edf_signals  #annotations

    X[i_run,:signals.shape[0],:] = signals

    #print(X.shape)

    # read annotations
    current_event = [i_run, 0, 0, 0] # run, class (l/r), start, end
    print(current_event)

    for annotation in annotations:
        t = int(annotation[0] * SAMPLE_RATE * 1e-7)
        action = int(annotation[2][1])
        print(action)




