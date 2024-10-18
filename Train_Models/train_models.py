import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from constants import TRAINING_FOLDER, ML_MODEL_PARAMS_FILE


def get_model_params(emb):
    params = json.loads(ML_MODEL_PARAMS_FILE) 
    return params[emb]


def vector_combination(emb, pairs, vector_operation, vector_size):
    x, y = [], []
    for prot1, prot2, label in pairs:
        emb_prot1 = np.array(emb[prot1]).reshape(1, vector_size)
        emb_prot2 = np.array(emb[prot2]).reshape(1, vector_size)
        if vector_operation == "Hadamard":
            op = np.multiply(emb_prot1, emb_prot2)
        elif vector_operation == "Concatenation":
            op = np.concatenate((emb_prot1, emb_prot2), axis=1)
        elif vector_operation == "Average":
            op = np.mean((emb_prot1, emb_prot2), axis=0)
        elif vector_operation == "WL1": 
            op = np.abs(np.subtract(emb_prot1, emb_prot2))
        elif vector_operation == "WL2":
            op = np.square((np.subtract(emb_prot1, emb_prot2)))
        elif vector_operation == 'CosSim':
            op = metrics.pairwise.cosine_similarity(emb_prot1, emb_prot2)
        else:
            raise ValueError("Vector operation needs to be in ['Hadamard', 'Concatenation', 'Average', 'WL1', 'WL2', 'CosSim'].")
        x.append(op.flatten().tolist())
        y.append(int(label))
    if vector_operation == 'CosSim':
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)
    return x, y


def get_ml_data(embs, pairs, vec_op, vec_size):
    data = {}
    for id, emb in embs:
        x, y = vector_combination(emb, pairs, vector_operation=vec_op, vector_size=vec_size)
        data[(id, vec_op)] = (x, y)
    return data


def get_ml_stats(y_truth, y_preds, proba_preds):
    accuracy = metrics.accuracy_score(y_truth, y_preds)
    precision = metrics.precision_score(y_truth, y_preds)
    recall = metrics.recall_score(y_truth, y_preds)
    f1_noninteract, f1_interact = metrics.f1_score(y_truth, y_preds, average=None)
    weighted_avg_f1 = metrics.f1_score(y_truth, y_preds, average='weighted')
    if proba_preds is not None:
        fpr, tpr, _ = metrics.roc_curve(y_truth, proba_preds)
        return accuracy, precision, recall, f1_noninteract, f1_interact, weighted_avg_f1, fpr, tpr
    return accuracy, precision, recall, f1_noninteract, f1_interact, weighted_avg_f1


def train_ml_model(x, y, id, model, test_size, random_state, shuffle, trained_params=None):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state, shuffle=shuffle)
    if trained_params is not None:
        model.set_params(**get_model_params(id))
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba_preds = model.predict_proba(X_test)[::,1]
    return get_ml_stats(y_test, preds, proba_preds)


def run_embedding_training(data=None, embs=None, pairs=None, vec_op=None, vec_size=None, \
                           train_type='ml', threshold=None, threshold_search=None, model=RandomForestClassifier(), \
                            test_size=0.3, random_state=42, shuffle=True, trained_params=None):
    assert train_type in ['threshold', 'ml'], "Training type needs to be in ['threshold', 'ml']."
    assert (threshold_search is not None) or (threshold is not None) or (train_type == 'ml'), \
        "If train_type='threshold' and threshold_search=None, the threshold value needs to be explicitly provided."
    
    performance = {
        'Embedding Model': [],
        'Vector Operation': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 (class 0)': [],
        'F1 (class 1)': [],
        'F1 Average': [],
    }

    if data is None:
        assert not any(v is None for v in [embs, pairs, vec_op, vec_size]), \
            "If data=None, the embeddings and entity pairs need to be provided."
        data = [get_ml_data(embs, pairs, vec_op, vec_size)]

    if train_type == 'ml':
        plt.figure(figsize=(20, 5*(len(data[0])//3+1)))
        plt.suptitle("ROC curves for each model and vector operation", y=0.95, fontsize='xx-large')
        print("\nFitting ML models...")
        for emb_model in tqdm(data):
            for (i, ((id, vec_op), (x, y))) in enumerate(emb_model.items()):
                stats = train_ml_model(x, y, id, model, test_size, random_state, shuffle, trained_params)
                performance['Embedding Model'].append(id)
                performance['Vector Operation'].append(vec_op)
                performance['Accuracy'].append(stats[0])
                performance['Precision'].append(stats[1])
                performance['Recall'].append(stats[2])
                performance['F1 (class 0)'].append(stats[3])
                performance['F1 (class 1)'].append(stats[4])
                performance['F1 Average'].append(stats[5])
                plt.subplot(len(emb_model)//3+1, 3, i+1)
                plt.plot(stats[6], stats[7], label=f'{vec_op} | AUC = {metrics.auc(stats[6], stats[7]):.3f}')
                plt.plot(np.arange(0, 1.1, 0.1), 
                        np.arange(0, 1.1, 0.1), 
                        linestyle="dashed", 
                        linewidth=1, 
                        color="black")
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                plt.title(id)
                plt.legend()
        plt.savefig(TRAINING_FOLDER + 'embeddings_ROC_curves.png')
        pd.DataFrame(performance).sort_values(by=["F1 Average"], ascending=False).to_csv(TRAINING_FOLDER + "embedding_ml_performances.csv", float_format='%.4f', index=False)
    
    elif train_type == 'threshold':
        t_list = []
        plt.figure(figsize=(20, 5*(len(data[0])//3+1)))
        plt.suptitle("Thresholds vs F1 score for every embedding model", y=0.95, fontsize='xx-large')
        for emb_model in tqdm(data):
            for (i, ((id, vec_op), (x, y))) in enumerate(emb_model.items()):
                stats, best_thres, thres_vs_f1 = train_thresholds(x, y, id, threshold, threshold_search)
                performance["Embedding Model"].append(id)
                performance['Vector Operation'].append(vec_op)
                performance['Accuracy'].append(stats[0])
                performance['Precision'].append(stats[1])
                performance['Recall'].append(stats[2])
                performance['F1 (class 0)'].append(stats[3])
                performance['F1 (class 1)'].append(stats[4])
                performance['F1 Average'].append(stats[5])
                t_list.append(best_thres)
                plt.subplot(len(emb_model)//3+1, 3, i+1)
                plt.plot(thres_vs_f1.keys(), thres_vs_f1.values(), label=id)
                plt.annotate(text=round(best_thres, 2), 
                            xy=(best_thres, thres_vs_f1[best_thres]),
                            xytext=(best_thres, thres_vs_f1[best_thres]+0.02),
                            arrowprops=dict(arrowstyle='->'),
                            ha='center')
                plt.xlabel("Threshold")
                plt.ylabel("F1 score (average)")
                plt.title(id)
                plt.ylim(top=1)
        plt.savefig(TRAINING_FOLDER + 'embeddings_thres_vs_f1.png')
        performance = pd.DataFrame(performance)
        performance.insert(2, "Threshold", t_list)
        performance.sort_values(by=["F1 Average"], ascending=False).to_csv(TRAINING_FOLDER + "embeddings_threshold_performances.csv", float_format='%.4f', index=False)
    
    print("\nEmbedding model training completed.")
    

def threshold_searching(sims, y, metric, thresholds):
    assert (min(thresholds) >= 0 and max(thresholds) <= 1), "Thresholds need to be in the interval [0, 1]."
    best_f1 = -np.inf
    best_thres = None
    final_preds = None
    print(f"\nSearching for the best threshold for {metric}...")
    thres_vs_f1 = {}
    for thres in thresholds:
        preds = []
        for x in sims:
            if x[0] > thres:
                preds.append(1)
            else:
                preds.append(0)
        f1 = metrics.f1_score(y, preds, average='weighted')
        thres_vs_f1[thres] = f1
        if f1 > best_f1:
            best_f1 = f1
            best_thres = thres
            final_preds = preds
    print(f"Best threshold for {metric}: {best_thres:.2f}")
    return final_preds, best_thres, thres_vs_f1 


def train_thresholds(sims, y, metric, threshold, threshold_search):
    if threshold_search is not None:
        preds, best_thres, thres_vs_f1 = threshold_searching(sims, y, metric, threshold_search)
    else:
        thres_vs_f1 = None
        best_thres = threshold
        preds = []
        for x in sims:
            if x > threshold:
                preds.append(1)
            else:
                preds.append(0)
    stats = get_ml_stats(y, preds, proba_preds=None)
    return stats, best_thres, thres_vs_f1


def run_semantic_training(ssims, train_with_all=True, train_type='threshold', threshold=None, threshold_search=None, \
                          model=RandomForestClassifier(random_state=42), test_size=0.3, random_state=42, shuffle=True, trained_params=None):
    assert train_type in ['threshold', 'ml'], "Training type needs to be in ['threshold', 'ml']."
    assert (threshold_search is not None) or (threshold is not None) or (train_type == 'ml'), \
        "If train_type='threshold' and threshold_search=None, the threshold value needs to be explicitly provided."

    performance = {
            'Semantic Similarity Measure': [],
            "Threshold": [],
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1 (class 0)': [],
            'F1 (class 1)': [],
            'F1 Average': [],
        }
    
    smetrics = ssims.iloc[:,3:]
    smetrics_columns = [col for col in smetrics.columns]
    y = ssims["Label"]
    
    if train_type == 'threshold':
        plt.figure(figsize=(20,(len(smetrics_columns)//3)*4))
        plt.suptitle("Thresholds vs F1 score for every measure", y=0.95, fontsize='xx-large')
        print("\nTraining threshold-based prediction models with the semantic similarity measures...")
        i = 0
        prev_metric = None
        for metric in smetrics_columns:
            scaler = MinMaxScaler()
            ssims_scaled = scaler.fit_transform(ssims[metric].to_numpy().reshape(-1, 1))
            stats, best_thres, thres_vs_f1 = train_thresholds(ssims_scaled, y, metric, threshold, threshold_search)
            performance["Semantic Similarity Measure"].append(metric)
            performance["Threshold"].append(best_thres)
            performance['Accuracy'].append(stats[0])
            performance['Precision'].append(stats[1])
            performance['Recall'].append(stats[2])
            performance['F1 (class 0)'].append(stats[3])
            performance['F1 (class 1)'].append(stats[4])
            performance['F1 Average'].append(stats[5])
            if (metric.startswith(('Resnik', 'Lin', 'Jiang-Conrath')) and metric.split(" ")[0] != prev_metric) or (not metric.startswith(('Resnik', 'Lin', 'Jiang-Conrath'))):
                i += 1
                prev_metric = metric.split(" ")[0]
                plt.subplot(len(smetrics_columns)//6+1, 3, i)
            plt.plot(thres_vs_f1.keys(), thres_vs_f1.values(), label=metric)
            plt.annotate(text=round(best_thres, 2), 
                        xy=(best_thres, thres_vs_f1[best_thres]),
                        xytext=(best_thres, thres_vs_f1[best_thres]+0.02),
                        arrowprops=dict(arrowstyle='->'),
                        ha='center')
            plt.xlabel("Threshold")
            plt.ylabel("F1 score (average)")
            plt.ylim(top=1)
            plt.title(prev_metric)
            plt.legend()
        if thres_vs_f1 is not None:
            plt.savefig(TRAINING_FOLDER + "semantic_thres_vs_f1.png")
        pd.DataFrame(performance).sort_values(by=["F1 Average"], ascending=False).to_csv(TRAINING_FOLDER + "semantic_threshold_performances.csv", float_format='%.4f', index=False)
    
    elif train_type == 'ml':
        if train_with_all:
             smetrics_columns += ['All']
        plt.figure(figsize=(20, (len(smetrics_columns)//3)*4))
        plt.suptitle("ROC curves for each measure", y=0.95, fontsize='xx-large')
        print("\nFitting ML models...")
        i = 0
        prev_metric = None
        for metric in smetrics_columns:
            if metric == 'All':
                scaler = MinMaxScaler()
                x = scaler.fit_transform(smetrics)
            else:
                x = ssims[metric].to_numpy().reshape(-1, 1)
            y = ssims["Label"].to_numpy().reshape(-1, 1)
            stats = train_ml_model(x, y, metric, model, test_size, random_state, shuffle, trained_params)
            performance["Semantic Similarity Measure"].append(metric)
            performance['Accuracy'].append(stats[0])
            performance['Precision'].append(stats[1])
            performance['Recall'].append(stats[2])
            performance['F1 (class 0)'].append(stats[3])
            performance['F1 (class 1)'].append(stats[4])
            performance['F1 Average'].append(stats[5])
            if (metric.startswith(('Resnik', 'Lin', 'Jiang-Conrath')) and metric.split(" ")[0] != prev_metric) or (not metric.startswith(('Resnik', 'Lin', 'Jiang-Conrath'))):
                i += 1
                prev_metric = metric.split(" ")[0]
                plt.subplot(len(smetrics_columns)//6+1, 3, i)
            plt.plot(stats[6], stats[7], label=f'{metric} | AUC={metrics.auc(stats[6], stats[7]):.3f}')
            plt.plot(np.arange(0, 1.1, 0.1), 
                    np.arange(0, 1.1, 0.1), 
                    linestyle="dashed", 
                    linewidth=1, 
                    color="black")
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.title(f'{metric.split(" ")[0]}')
            plt.legend(loc='lower right')
        plt.savefig(TRAINING_FOLDER + 'semantic_ROC_curves.png')
        performance.pop('Threshold')
        pd.DataFrame(performance).sort_values(by=["F1 Average"], ascending=False).to_csv(TRAINING_FOLDER + "semantic_ml_performances.csv", float_format='%.4f', index=False)
    
    print("\nSemantic model training completed.")
