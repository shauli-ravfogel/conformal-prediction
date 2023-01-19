from transformers import AutoModel, AutoTokenizer
# import gpt2 LM
from transformers import AutoModelForCausalLM, GPT2Tokenizer
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
import gc
import matplotlib.pyplot as plt
import seaborn as sn

def load_gpt2_model(model_name):
    # load the gpt2 model and tokenizer
    # return: a pretrained model and a tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def get_adaptive_calibration_scores(all_probs, all_gold_token_ids):
    scores = np.zeros(all_probs.shape[0])
    # sum up all probabilities that are higher than the gold probability
    for i in range(all_probs.shape[0]):
        probs_i = all_probs[i]
        gold_token_id = all_gold_token_ids[i]
        gold_prob = probs_i[gold_token_id]
        scores[i] = np.sum(probs_i[probs_i >= (gold_prob - 1e-9)])
    
    return scores

def get_adaptive_qhat(scores, alpha):
    n = scores.shape[0]
    q_level = np.ceil((n+1)*(1-alpha))/n
    qhat = np.quantile(scores, q_level, method='higher')
    return qhat

def get_entropy_bins(entropies, all_probs, all_gold_token_ids, all_sents, all_positions):

    bin2probs = dict()
    bin2gold_token_ids = dict()
    bin2sent = dict()
    bin2position = dict()

    # calcualte the entropy bins. use the pre-calculated entropy values in entropies
    num_bins = 10
    bin_size = int(len(entropies)/num_bins)
    gc. collect()
    # sort the entropies and sort the all_probs and all_gold_token_ids accordingly
    sorted_entropies_idx = np.argsort(entropies)
    sorted_probs = all_probs[sorted_entropies_idx,:]
    sorted_gold_token_ids = all_gold_token_ids[sorted_entropies_idx]
    entropies_sorted = entropies[sorted_entropies_idx]
    # sorted all_sentences:
    all_sents_sorted = [all_sents[i] for i in sorted_entropies_idx]
    all_positions_sorted = [all_positions[i] for i in sorted_entropies_idx]

    # split the sorted_probs and sorted_gold_token_ids into num_bins chunks
    for i in range(num_bins):
        bin2probs[i] = sorted_probs[i*bin_size:(i+1)*bin_size]
        bin2gold_token_ids[i] = sorted_gold_token_ids[i*bin_size:(i+1)*bin_size]
        bin2sent[i] = all_sents_sorted[i*bin_size:(i+1)*bin_size]
        bin2position[i] = all_positions_sorted[i*bin_size:(i+1)*bin_size]

    return bin2probs, bin2gold_token_ids, bin2sent, bin2position


def calibrate_per_entropy_bins(bin2probs, bin2gold_token_ids):

    bin2qhat = defaultdict(dict)
    bin2cal_scores = defaultdict(dict)
    bin2prediction_sets = defaultdict(dict)
    num_bins = len(bin2probs)
    
    for i in range(num_bins):
        cal_scores = get_adaptive_calibration_scores(bin2probs[i], bin2gold_token_ids[i])
        for alpha in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]:
            qhat = get_adaptive_qhat(cal_scores, alpha)
            bin2qhat[alpha][i] = qhat
            bin2cal_scores[alpha][i] = cal_scores
    
    return bin2qhat, bin2cal_scores

def plot_overall_calibration(alpha_vals, qhats, model_name):
    
    sn.set_style("darkgrid")
    plt.figure(figsize=(10,6))
    # plot 1-alpha (x axis) vs qhat (y axis).
    
    plt.plot(1-np.array(alpha_vals), qhats, label="$\hat{q}$ values")
    # for the x label, use the latex of the alpha symbol
    plt.xlabel("$1-\\alpha$ (Confidence)", fontsize=17)
    # for the y label, use the latex of the qhat symbol rotated 90 degrees
    plt.ylabel("$\hat{q}$", rotation=0, fontsize=17)
    # start from (0,0)
    plt.xlim(0,1)
    plt.ylim(0,1)

    # plot a dashed diagonal for y=x
    plt.plot([0.0,1],[0.0,1], linestyle='--', color='black', label="Calibrated model")
    # increase x,y font size as well as the legend font size
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)

    plt.savefig("adaptive_calibration_{}_q_vs_alpha.pdf".format(model_name.replace("facebook/", "")), dpi=700)
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


def get_effective_acc(sorted_probs, sorted_probs_idx,  gold_token_ids_bin, qhat):
        accuracies = []
        for i in range(len(sorted_probs_idx)):
        
            # find an index j such that the sum of the probs until index j is larger or equal to qhat.
            for j in range(len(all_probs_sorted[i])):
                if np.sum(sorted_probs[i][:j]) >= qhat:
                    threshold = j
                    break
            top_qhat_probs = sorted_probs_idx[i,:threshold]
            gold_token_id = gold_token_ids_bin[i]
            if gold_token_id in top_qhat_probs:
                accuracies.append(1)
            else:
                accuracies.append(0) 
        return np.mean(accuracies), accuracies

def effective_acc_vs_threshold(sorted_probs, sorted_probs_idx, all_gold_token_ids):
    qhats = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    qhat2avg_accuracy = dict()
    for qhat in qhats:
        accuracies = []
        for i in range(len(sorted_probs_idx)):
        
            # find an index j such that the sum of the probs until index j is larger or equal to qhat.
            for j in range(len(all_probs_sorted[i])):
                if np.sum(sorted_probs[i][:j]) >= qhat:
                    threshold = j
                    break
            top_qhat_probs = sorted_probs_idx[i,:threshold]
            gold_token_id = all_gold_token_ids[i]
            if gold_token_id in top_qhat_probs:
                accuracies.append(1)
            else:
                accuracies.append(0)
        qhat2avg_accuracy[qhat] = accuracies
    return qhat2avg_accuracy

def plot_entropy_percentiles(bin2qhat, model_name):
    import matplotlib.pyplot as plt
    # increae the plot to prevent the x axis labels from being cut off
    sn.set_style("darkgrid")
    plt.figure(figsize=(10,6))
    num_bins = len(bin2qhat[0.1])
    # plot the qhat vs bins for each alpha value
    percentiles = np.arange(0, 100 + 100/num_bins, 100/num_bins)
    #plt.plot(percentiles[1:], list(bin2qhat.values()))
    for alpha in bin2qhat.keys():
        plt.plot(percentiles[1:], list(bin2qhat[alpha].values()), label="$1-\\alpha$={}".format(1-alpha))
    plt.xlabel("Entropy Percentile", fontsize=17)
    plt.ylabel("$\hat{q}$", rotation=0, fontsize=17)
    plt.title("$\hat{q}$ vs Entropy Percentile", fontsize=17)
    # increase x,y font size as well as the legend font size
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    #plt.legend()
    plt.savefig("adaptive_calibration_{}_q_vs_entropy_percentile.pdf".format(model_name.replace("facebook/", "")), dpi=700)
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

def extract_model_size(model_name):
    model_size = model_name.split("-")[1]
    if model_size.endswith("m"):
        model_size = int(float(model_size[:-1]) * 1000000)
    elif model_size.endswith("b"):
        model_size = int(float(model_size[:-1]) * 1000000000)
    return model_size

model_names = ["facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b" ,"facebook/opt-6.7b", "facebook/opt-13b", "facebook/opt-30b"]
model_names = ["facebook/opt-350m"]#, "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b" ,"facebook/opt-6.7b", "facebook/opt-13b", "facebook/opt-30b"]
model_size2qhat = dict()
model_size2bin2qhat = dict()
model_size2qhat2effective_acc = dict()
model_size2bin2effective_acc = defaultdict(dict)
use_1_per_sent = False

for model_name in model_names:
    print("Working on model {}".format(model_name))
    with open("preds_10000_{}.pickle".format(model_name.replace("facebook/", "")), "rb") as f:
        preds = pickle.load(f)

    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loaded tokenizker")

    # calculate the entropies

    entropies = []
    all_probs = []
    all_gold_tokens = []
    all_gold_token_ids =[]
    all_sents = []
    all_positions = []

    for p_dict in preds:
        probs = p_dict["probs"]
        probs = probs[:-1,:] # remove the last token
        entropy = (-np.sum(probs * np.log(probs), axis=1)).tolist()
    
        token_ids = p_dict["tokens"][1:] # remove start of sentence token
        positions = list(range(len(entropy)))
        all_positions.extend(positions)
        entropies.extend(entropy)
        all_probs.extend(probs)
        all_gold_token_ids.extend(token_ids)
        # get the ids from the tokens
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        all_gold_tokens.extend(tokens)
        for i in range(len(tokens)):
            all_sents.append(p_dict["sent"])

    print("Collected entropies")

    all_gold_token_ids = np.array(all_gold_token_ids)
    all_gold_tokens = np.array(all_gold_tokens)
    all_probs = np.array(all_probs)
    entropies = np.array(entropies)
    all_positions = np.array(all_positions)

    # shuffle all
    from sklearn.utils import shuffle
    all_gold_token_ids, all_gold_tokens, all_probs, entropies, all_sents, all_positions = shuffle(all_gold_token_ids, all_gold_tokens, all_probs, entropies, all_sents, all_positions)
    if use_1_per_sent:
        # from each sentence, keep only 1 example
        idx_to_keep = []
        sent2idx = dict()
        for i in range(len(all_sents)):
            if all_sents[i] not in sent2idx:
                sent2idx[all_sents[i]] = i
                idx_to_keep.append(i)
        all_gold_token_ids = all_gold_token_ids[idx_to_keep]
        all_gold_tokens = all_gold_tokens[idx_to_keep]
        all_probs = all_probs[idx_to_keep]
        entropies = entropies[idx_to_keep]
        all_sents = [s for i, s in enumerate(all_sents) if i in idx_to_keep]
        all_positions = all_positions[idx_to_keep]


    # calibration -- all instances

    alpha_vals = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    qhats = []
    scores = get_adaptive_calibration_scores(all_probs, all_gold_token_ids)
    for alpha in alpha_vals:
        print("Calibrating for alpha = {}".format(alpha))
        qhat = get_adaptive_qhat(scores, alpha)
        qhats.append(qhat)
    
    
    model_size2qhat[extract_model_size(model_name)] = qhats
    print("Plotting....")
    plot_overall_calibration(alpha_vals, qhats, model_name)
    print("Getting entropy bins...")

    # calibration -- per entropy bin

    bin2probs, bin2gold_token_ids, bin2sent, bin2position = get_entropy_bins(entropies, all_probs, all_gold_token_ids, all_sents, all_positions)
    print("Caliberating per entropy bin...")
    bin2qhat, bin2cal_scores = calibrate_per_entropy_bins(bin2probs, bin2gold_token_ids)
    model_size2bin2qhat[extract_model_size(model_name)] = bin2qhat
    print("Plotting...")
    plot_entropy_percentiles(bin2qhat, model_name)

    
    # threshold vs effective accuracy
    
    print("Calculating threshold vs effective accuracy")
    all_probs_argsorted = np.argsort(all_probs, axis=1)[:,::-1]
    all_probs_sorted = np.sort(all_probs, axis=1)[:,::-1]
    qhat2accs = effective_acc_vs_threshold(all_probs_sorted, all_probs_argsorted, all_gold_token_ids)
    model_size2qhat2effective_acc[extract_model_size(model_name)] = qhat2accs

    #threshold per entropy bin 
    for bin in bin2probs:
        print("Calculating effective accuracy for bin {}".format(bin))
        probs = bin2probs[bin]
        probs_argsorted = np.argsort(probs, axis=1)[:,::-1]
        probs_sorted = np.sort(probs, axis=1)[:,::-1]
        gold_token_ids_bin = bin2gold_token_ids[bin]
        acc, accs = get_effective_acc(all_probs_sorted, probs_argsorted, gold_token_ids_bin, qhat=0.9)
        print(np.mean(accs))
        model_size2bin2effective_acc[extract_model_size(model_name)][bin] = accs
    
    prefix = "1_per_sent" if use_1_per_sent else ""
    with open("model_size2qhat2effective_acc{}_v2.pickle".format(prefix), "wb") as f:
         pickle.dump(model_size2qhat2effective_acc, f)
    with open("model_size2qhat{}_v2.pickle".format(prefix), "wb") as f:
       pickle.dump(model_size2qhat, f)
    with open("model_size2bin2qhat{}_v2.pickle".format(prefix), "wb") as f:
       pickle.dump(model_size2bin2qhat, f)
    with open("model_size2bin2effective_acc{}_v2.pickle".format(prefix), "wb") as f:
          pickle.dump(model_size2bin2effective_acc, f)
