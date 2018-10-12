from CpG_Net import CpGNet
from CpG_Bin import Bin
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from random import shuffle
from sklearn.metrics import roc_curve, auc
import sys


CPG_DENSITY = (int)(sys.argv[1])


# Creates the accuracy/confidence/percent imputed plots
def get_tradeoff(X_data, pred, y_actual, model, model_name):
    confs = []
    accs = []
    totals_perc = []
    for conf in ([0.01]+[x/10.0 for x in range(1,10)] + [0.99]):
        thresh=conf/2.0
        num_correct = 0
        num_total = 0
        for i in range(pred.shape[0]):
            #print "distance:",np.abs(0.5-pred[i])
            if np.abs(0.5-pred[i]) > thresh:
                num_total+=1
                if np.round(pred[i]) == y_actual[i]:
                    num_correct +=1
        if num_total != 0:
            acc = num_correct/float(num_total)
            confs.append(conf)
            accs.append(acc)
            totals_perc.append(num_total/float(len(pred)))

    fig, ax1 = plt.subplots()
    plt.title("Accuracy-Data Gained Tradeoff for "+ model_name, fontsize=15)
    plt.axhline(y=0.95, color='r', linestyle='--')
    
    plt.ylim([0,1])

    ax2 = ax1.twinx()
    ax1.plot(confs, accs, 'g-')

    ax2.plot(confs, totals_perc, 'b-')
    ax1.set_xlabel('Confidence',fontsize=15)
    ax1.set_ylabel('Accuracy', color='g',fontsize=15)
    ax2.set_ylabel('Percent Imputed', color='b',fontsize=15)
    plt.ylim([0,1])
    plt.savefig(str(CPG_DENSITY)+"tradeoff"+model_name+".png",dpi=500)
    plt.show() 


print "loading data"
data = pickle.load(open("HAMbins.p","rb")) 

min_read_depth = 20
read_filtered_data = [bin_ for bin_ in data if bin_.matrix.shape[0] >= min_read_depth]


cpg_2_bins = [bin_ for bin_ in read_filtered_data if bin_.matrix.shape[1]==2]
cpg_3_bins = [bin_ for bin_ in read_filtered_data if bin_.matrix.shape[1]==3]
cpg_4_bins = [bin_ for bin_ in read_filtered_data if bin_.matrix.shape[1]==4]
cpg_5_bins = [bin_ for bin_ in read_filtered_data if bin_.matrix.shape[1]==5]
cpg_6_bins = [bin_ for bin_ in read_filtered_data if bin_.matrix.shape[1]==6]
cpg_7_bins = [bin_ for bin_ in read_filtered_data if bin_.matrix.shape[1]==7]
cpg_8_bins = [bin_ for bin_ in read_filtered_data if bin_.matrix.shape[1]==8]

bin_dict = {}
bin_dict[2] = cpg_2_bins
bin_dict[3] = cpg_3_bins
bin_dict[4] = cpg_4_bins
bin_dict[5] = cpg_5_bins
bin_dict[6] = cpg_6_bins
bin_dict[7] = cpg_7_bins
bin_dict[8] = cpg_8_bins



# get a subset of the data to speed things up
cpgs_bins = bin_dict[CPG_DENSITY]
shuffle(cpgs_bins)
cpgs_bins_subset = cpgs_bins

net = CpGNet(cpgDensity=CPG_DENSITY)
print "collecting Features"

X, y = net.collectFeatures(cpgs_bins_subset) # extract features

# filter out missing values
nonneg = y!=-1
X_u = X[nonneg]
y_u = y[nonneg]
X_train, X_test, y_train, y_test = train_test_split(X_u, y_u)

print "preprocessing"

X_train_scaled = preprocessing.scale(X_train)
X_test_scaled = preprocessing.scale(X_test)

print "training"

history = net.fit(X_train_scaled, y_train, val_split=0.2, epochs=1000, batch_size=16)

plt.plot(history.history["loss"],label="training")
plt.plot(history.history["val_loss"],label="validation")
#plt.ylim([0,0.4])
plt.xlabel("Training Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Model Loss")
plt.savefig(str(CPG_DENSITY)+"model_loss.png",dpi=500)
plt.show()


plt.cla()
plt.plot(history.history["acc"],label="training")
plt.plot(history.history["val_acc"],label="validation")
#plt.ylim([0,1])
plt.xlabel("Training Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Model Accuracy")
plt.savefig(str(CPG_DENSITY)+"model_acc.png",dpi=500)

plt.show()
plt.cla()



y_train_pred = net.predict(X_train_scaled)
y_train_pred_bin = np.copy(y_train_pred)
y_train_pred_bin[y_train_pred_bin>0.5] = 1
y_train_pred_bin[y_train_pred_bin<0.5] = 0

accuracy_score(y_train_pred_bin, y_train)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_train_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print "AUC:",roc_auc
plt.title("Train ROC, AUC = %0.5f"% roc_auc, fontsize=15)
plt.plot(false_positive_rate, true_positive_rate, 'b')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate',fontsize=15)
plt.xlabel('False Positive Rate',fontsize=15)

plt.savefig(str(CPG_DENSITY)+"roc_train.png",dpi=500)


plt.cla()
get_tradeoff(X_train_scaled, y_train_pred, y_train, net, "CpG Net Train")
plt.cla()
y_test_pred = net.predict(X_test_scaled)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_test_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print "AUC:",roc_auc
plt.title("Test Data ROC, AUC = %0.5f"% roc_auc, fontsize=15)
plt.plot(false_positive_rate, true_positive_rate, 'b')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate',fontsize=15)
plt.xlabel('False Positive Rate',fontsize=15)
plt.savefig(str(CPG_DENSITY)+"roc_test.png",dpi=500)
plt.cla()
get_tradeoff(X_test_scaled, y_test_pred, y_test, net, "CpG Net Test")
plt.cla()
print "done"


