import numpy as np
import torch
from torcheval.metrics import BinaryPrecision, BinaryRecall, BinaryF1Score
from sklearn.metrics import classification_report, ndcg_score, jaccard_score
from utils.utils import pdist, query_dataset_dist, query_dataset_sim_cosine
from datasets.base import LabelVectorHelper


class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

class ClassificationReport(Metric):
    def __init__(self, proximity_vector_labels_dict, classes_list, k=[1, 5, 10]):
        self.pred = None
        self.target = None
        self.proximity_vector_labels_dict = proximity_vector_labels_dict
        self.classes_list = classes_list
        self.k = k
        self.report = ''

    def reset(self):
        self.pred = None
        self.target = None
        self.reports = ''
        
    def value(self):
        return self.report

    def name(self):
        return 'Classification Report'
    
    def __call__(self, outputs, target, loss, n_triplets_list):
        distances_matrix = pdist(outputs)
        sorted_args = np.argsort(distances_matrix)
        sorted_args = sorted_args[:, 1:].numpy() # remove first column because it is the self-distance (always zero)
        target = target.numpy()
        nearest_neigbors_as_class_vectors = np.array([self.proximity_vector_labels_dict[target[a]] for a in sorted_args])
        targets_as_class_vectors = np.array([self.proximity_vector_labels_dict[t] for t in target])

        self.report = classification_report(target, outputs, target_names=self.classes_list, digits=4, output_dict=True)
        return self.value()

class BinaryAccuracyMetric(Metric):
    def __init__(self):
        self.correct = 0
        self.total = 0
        self.metrics_dict = dict()

    def reset(self):
        self.correct = 0
        self.total = 0
        self.metrics_dict = dict()

    def value(self):
        values = dict()
        for (k, v) in self.metrics_dict.items():
            values[k] = round(v.compute().item(), 4)
        return values

    def name(self):
        return 'Per-class Accuracy'

    def __call__(self, pred, target, class_names):
        self.metrics_dict = {cname: BinaryPrecision() for cname in class_names}
        for i, cname in enumerate(class_names):
            class_pred = pred[:, i]
            class_target = target[:, i]
            self.metrics_dict[cname].update(class_pred, class_target)
        return self.value()
    

class BinaryRecallMetric(Metric):
    def __init__(self):
        self.correct = 0
        self.total = 0
        self.metrics_dict = dict()

    def reset(self):
        self.correct = 0
        self.total = 0
        self.metrics_dict = dict()

    def value(self):
        values = dict()
        for (k, v) in self.metrics_dict.items():
            values[k] = round(v.compute().item(), 4)
        return values

    def name(self):
        return 'Per-class Recall'

    def __call__(self, pred, target, class_names):
        self.metrics_dict = {cname: BinaryRecall() for cname in class_names}
        for i, cname in enumerate(class_names):
            class_pred = pred[:, i]
            class_pred = torch.round(class_pred).to(torch.uint8)
            class_target = target[:, i]
            class_target = class_target.to(torch.uint8)
            self.metrics_dict[cname].update(class_pred, class_target)
        return self.value()
    
class BinaryF1ScoreMetric(Metric):
    def __init__(self):
        self.correct = 0
        self.total = 0
        self.metrics_dict = dict()

    def reset(self):
        self.correct = 0
        self.total = 0
        self.metrics_dict = dict()

    def value(self):
        values = dict()
        for (k, v) in self.metrics_dict.items():
            values[k] = round(v.compute().item(), 4)
        return values

    def name(self):
        return 'Per-class F1-Score'

    def __call__(self, pred, target, class_names):
        self.metrics_dict = {cname: BinaryF1Score() for cname in class_names}
        for i, cname in enumerate(class_names):
            class_pred = pred[:, i]
            class_target = target[:, i]
            self.metrics_dict[cname].update(class_pred, class_target)
        return self.value()
    


class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0
        self.print_counter = 0

    def __call__(self, outputs, target, loss):
        self.print_counter += 1
        
        #pred = outputs[0].data.max(1, keepdim=True)[1]
        pred = outputs[0].data
        target = target[0]
        #print('outputs:', outputs)
        #print('outputs[0].data:', outputs[0].data)
        #print('outputs[0].data.max(1, keepdim=True):', outputs[0].data.max(1, keepdim=True))
        #print('outputs[0].data.max(1, keepdim=True)[1]:', outputs[0].data.max(1, keepdim=True)[1])
        #print('outputs[0]:', outputs[0])
        #print('target[0].size()[0]:', target[0].size()[0])
        quantized_pred = torch.round(pred).to(torch.uint8) 
        accs = torch.tensor([(p == t).all() for p, t in zip(quantized_pred, target.to(torch.uint8))])
        #accs = torch.mean(accs, 1)
        #accs = np.ones(target[0].size()[0]) - np.abs(pred.cpu().numpy() - target[0].cpu().numpy()).mean(axis=1)
        #print('accs:', accs)
        self.correct += accs.sum().item()
        #self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target.size(0)

        if self.print_counter % 100 == 1:
            print('pred.shape:', pred.size())
            print('target.shape:', target.size())
            print('pred:', pred)
            print('target:', target)
            print('correct:', self.correct)
            print('total:', self.total)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'


class AverageNonzeroTripletsMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []

    def __call__(self, epoch_number, dataset_embeddings, dataset_labels, query_embeddings, query_labels, loss, n_triplets_list, tensorboard_writer, training=True):
        #print('loss:', loss)
        self.values = n_triplets_list
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return round(np.mean(self.values), 1)

    def name(self):
        return 'Average nonzero triplets'
    
class TotalNonzeroTripletsMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []

    def __call__(self, epoch_number, dataset_embeddings, dataset_labels, query_embeddings, query_labels, loss, n_triplets_list, tensorboard_writer, training=True):
        #print('loss:', loss)
        self.values = n_triplets_list
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.sum(self.values)

    def name(self):
        return 'Total nonzero triplets'


class Loss(Metric):
    '''
    Helper class to log and report the mean loss of a model evaluation.
    '''

    def __init__(self):
        self.loss = -1.

    def __call__(self, epoch_number, dataset_embeddings, dataset_labels, query_embeddings, query_predicted_logits, query_labels, loss, n_triplets_list, tensorboard_writer, training=True):
        _ = dataset_embeddings
        _ = dataset_labels
        _ = query_embeddings
        _ = query_predicted_logits
        _ = query_labels
        self.loss = loss
        _ = n_triplets_list
        if training:
            log_name_prefix = 'training'
        else:
            log_name_prefix = 'validation'
        metric_name = f'{log_name_prefix}_loss'
        tensorboard_writer.add_scalar(metric_name, self.value(), epoch_number)
        return self.value()

    def reset(self):
        self.loss = -1.

    def value(self):
        return round(self.loss, 4)
        
    def name(self):
        return f'Mean Loss'

class NDCG(Metric):
    '''
    Calculates the Normalized Discounted Cumulative Gain in a set of items.
    Each item is interpreted as a query against the rest of the set.
    Relevance scores are calculated from each item's labels
    '''

    def __init__(self, proximity_vector_labels_dict, classes_list, k=[10]):
        self.proximity_vector_labels_dict = proximity_vector_labels_dict
        self.classes_list = classes_list
        self.n_classes = len(self.classes_list)
        self.relevance_orders = self.get_label_relevance_orders()
        self.rng = np.random.default_rng(seed=0)
        self.k = k
        self.ndcg_value = None
        self.ndcg_at_k = []
        self.ndcg_at_k_random = [] 
        self.per_class_ndcg_scores = []
        self.ndcg_per_class = dict()
        self.ndcg_aggregated = dict()

    def get_query_per_class_relevance(self, query, result):
        query = np.array(query)
        result = np.array(result)
        rel = (query & result).astype(float)
        return rel

    def get_query_relevance(self, query, result):
        query = np.array(query)
        result = np.array(result)
        # rel = query[query == result]
        # if sum(rel) == 0  and len(rel) == self.n_classes:
        #     # query and result are in the "none" class i.e. their labels are all zeros
        #     # in this case, assign a relevance of 1.
        #     rel_score = 1.0
        # elif len(rel) == 0:
        #     # no elements in common between query and result
        #     # relevance is zero in this case
        #     rel_score = 0.0
        # else:
        #     rel_score = sum(rel)/max(sum(query), sum(result))
        # return rel_score
        return jaccard_score(query, result, zero_division=1.0)

    def get_label_relevance_orders(self):
        inter_label_relevance_dict = dict()
        for label_id, label_vector in self.proximity_vector_labels_dict.items():
            inter_label_relevance_dict[label_id] = [
                self.get_query_relevance(
                    label_vector, 
                    self.proximity_vector_labels_dict[r]
                ) for r in self.proximity_vector_labels_dict
             ]
        ordered_relevance_indices_dict = {
            label_id: list(np.flip(np.argsort(inter_label_relevance_dict[label_id]))) for label_id in inter_label_relevance_dict
        }
        return ordered_relevance_indices_dict

    def per_class_relevance_metric(self, query_label, results_list):
        query_vector = self.proximity_vector_labels_dict[query_label]
        query_results_list = [self.proximity_vector_labels_dict[r] for r in results_list]
        query_relevances = [self.get_query_per_class_relevance(query_vector, result_vector) for result_vector in query_results_list]
        return query_relevances

    def relevance_metric(self, query_label, results_list):
        query_vector = self.proximity_vector_labels_dict[query_label]
        query_results_list = [self.proximity_vector_labels_dict[r] for r in results_list]
        query_relevances = [self.get_query_relevance(query_vector, result_vector) for result_vector in query_results_list]
        return query_relevances
    
    def calculate_per_class_ndcg(self, query_label, relevances, ideal_relevances):
        k_top = [min(k, len(relevances)) for k in self.k]
        query_vector = self.proximity_vector_labels_dict[query_label]
        dcg_at_k = np.array([relevances[i]/np.log2((i+1)+1) for k in k_top for i in range(k)])
        ideal_dcg_at_k = np.array([ideal_relevances[i]/np.log2((i+1)+1) for k in k_top for i in range(k)])
        per_class_ndcg = dict()
        for i, class_is_queried in enumerate(query_vector):
            if class_is_queried == 0:
                continue
            current_class = self.classes_list[i]
            per_class_ndcg[current_class] = {k_original: np.sum(dcg_at_k[k, i])/max(1., np.sum(ideal_dcg_at_k[k, i])) for k, k_original in enumerate(self.k)}
        return per_class_ndcg

    def calculate_ndcg(self, relevances, ideal_relevances):
        dcg = [relevances[i]/np.log2((i+1)+1) for i in range(len(relevances))]
        ideal_dcg = [ideal_relevances[i]/np.log2((i+1)+1) for i in range(len(ideal_relevances))]
        k_top = [min(k, len(dcg)) for k in self.k]
        ndcg_at_k = {k_original: np.sum(dcg[:k])/np.sum(ideal_dcg[:k]) for k, k_original in zip(k_top, self.k)}
        return ndcg_at_k

    def __call__(self, epoch_number, dataset_embeddings, dataset_labels, query_embeddings, query_labels, loss, n_triplets_list, tensorboard_writer, training=True):
        print('Call on Metric.NDCG')
        distances_matrix = query_dataset_dist(query_embeddings, dataset_embeddings)
        sorted_args = np.argsort(distances_matrix) # indices of distances sorted in increasing order
        if training:
            # during training, query is the same as the dataset 
            # In this case, remove first column of the sorted matrix because it corresponds to the self-distance (always zero)
            sorted_args = sorted_args[:, 1:].numpy() 
        dataset_labels = dataset_labels.numpy()
        query_labels = query_labels.numpy()
        '''
        results_label_vectors = np.array([[self.proximity_vector_labels_dict[target[i]] for i in row] for row in sorted_args])
        ideal_results_args = np.array([sorted(row, key=lambda v: self.relevance_orders[target[i]].index(target[v])) for i, row in enumerate(sorted_args)])
        ideal_results_label_vectors = np.array([[self.proximity_vector_labels_dict[target[i]] for i in row] for row in ideal_results_args])
        self.ndcg_scores = [ndcg_score(ideal_result, result, k=self.k) for ideal_result, result in zip(ideal_results_label_vectors, results_label_vectors)]
        return self.value()
        '''
        for i, row in enumerate(sorted_args):
            #row = row.cpu().numpy().tolist()
            current_query_label = query_labels[i].item()
            results_labels = [dataset_labels[i].item() for i in row]
            relevance_scores = self.per_class_relevance_metric(current_query_label, results_labels)
            relevance_scores_random = self.rng.choice(relevance_scores, size=len(relevance_scores), replace=False)
            #relevance_scores = self.relevance_metric(current_query_label, results_labels)
            #relevance_scores.append(relevance_row)
            results_ideal_scores_order = sorted(row, key=lambda v: self.relevance_orders[current_query_label].index(dataset_labels[v]))
            results_ideal_scores_labels = [dataset_labels[i].item() for i in results_ideal_scores_order]
            ideal_relevance_scores = self.relevance_metric(current_query_label, results_ideal_scores_labels)
            print('Calculating ndcg_at_k')
            self.ndcg_at_k.append(self.calculate_ndcg(relevance_scores, ideal_relevance_scores))
            print('Calculating ndcg_at_k_random')
            self.ndcg_at_k_random.append(self.calculate_ndcg(relevance_scores_random, ideal_relevance_scores))
            per_class_ideal_relevance_scores = self.per_class_relevance_metric(current_query_label, results_ideal_scores_labels)
            print('Calculating per_class_ndcg_scores')
            self.per_class_ndcg_scores.append(self.calculate_per_class_ndcg(current_query_label, relevance_scores, per_class_ideal_relevance_scores))
            
            '''
            #results_vectors = np.array([self.proximity_vector_labels_dict[l] for l in row_labels])
            #ideal_results_vectors = np.array([self.proximity_vector_labels_dict[l] for l in row_ideal_scores_labels_order])
            #sklearn_ndcg_scores = {class_name: {k: [] for k in self.k} for class_name in self.classes_list}
            #for j, class_name in enumerate(self.classes_list):
            #    for k_value in self.k:
            #        sklearn_ndcg_scores[class_name][k_value].append(ndcg_score(ideal_results_vectors[:, j], results_vectors[:, j], k=k_value))
            #self.ndcg_scores = sklearn_ndcg_scores
            '''

        self.ndcg_value = self.value()
        if training:
            log_name_prefix = 'training'
        else:
            log_name_prefix = 'validation'
        for ndcg_name in self.ndcg_value:
            for k in self.ndcg_value[ndcg_name]:
                metric_name = f'{log_name_prefix}_{ndcg_name}@{k}'
                tensorboard_writer.add_scalar(metric_name, self.ndcg_value[ndcg_name][k], epoch_number)

        return self.ndcg_value



    def reset(self):
        self.ndcg_value = None
        self.ndcg_at_k = []
        self.ndcg_at_k_random = []
        self.per_class_ndcg_scores = []
        self.ndcg_per_class = dict()
        self.ndcg_aggregated = dict()

    def value(self):
        print('Call on Metric.NDCG.value()')
        # self.value() result is stored in the variable self.ndcg_value
        # If self.ndcg_value is already calculated, return it
        if self.ndcg_value is not None:
            return self.ndcg_value
        # calculate per query NDCG & NDCG_random
        ndcg = {k: 0.0 for k in self.k}
        ndcg_random = {k: 0.0 for k in self.k}
        query_count = 0
        for ndcg_scores, ndcg_random_scores in zip(self.ndcg_at_k, self.ndcg_at_k_random):
            query_count += 1
            for k_value in ndcg_scores:
                ndcg[k_value] += ndcg_scores[k_value]
                ndcg_random[k_value] += ndcg_random_scores[k_value]
        for k_value in ndcg:
            ndcg[k_value] /= query_count
            ndcg[k_value] = round(ndcg[k_value], 4)
            ndcg_random[k_value] /= query_count
            ndcg_random[k_value] = round(ndcg_random[k_value], 4)
        self.ndcg = ndcg
        self.ndcg_random = ndcg_random
        # calculate per class NDCG (query binary decomposition)
        class_query_count = dict()
        ndcg_reduced = dict()
        ndcg_aggregated = dict()
        total_queries = 0
        for ndcg_score in self.per_class_ndcg_scores:
            for class_name in ndcg_score:
                if class_name not in ndcg_reduced:
                    ndcg_reduced[class_name] = {k: 0.0 for k in self.k}
                if class_name not in class_query_count:
                    class_query_count[class_name] = 0
                class_query_count[class_name] += 1
                total_queries += 1
                for k_value in ndcg_score[class_name]:
                    ndcg_reduced[class_name][k_value] += ndcg_score[class_name][k_value]
        for class_name in ndcg_reduced:
            for k_value in ndcg_reduced[class_name]:
                ndcg_reduced[class_name][k_value] /= class_query_count[class_name]
                if k_value not in ndcg_aggregated:
                    ndcg_aggregated[k_value] = 0.0
                ndcg_aggregated[k_value] += ndcg_reduced[class_name][k_value] * (class_query_count[class_name]/total_queries)
                ndcg_reduced[class_name][k_value] = round(ndcg_reduced[class_name][k_value], 4)
        ndcg_aggregated = {k: round(v, 4) for k, v in ndcg_aggregated.items()}
        self.ndcg_aggregated = ndcg_aggregated
        self.ndcg_per_class = ndcg_reduced
        return dict([('NDCG', self.ndcg)]+[('NDCG_random', self.ndcg_random)]+[('NDCG_per_class_aggregated', self.ndcg_aggregated)]+[(f'NDCG_{cname}', self.ndcg_per_class[cname]) for cname in self.ndcg_per_class])
        

    def name(self):
        return f'NDCG@k: k={self.k}'

'''
class Recall(Metric):
    
    #Calculates the Recall in a set of items.
    #Each item is interpreted as a query against the rest of the set.
    #Matches are calculated from each item's labels
    

    def __init__(self, proximity_vector_labels_dict, k=[1, 5, 10, 25]):
        self.proximity_vector_labels_dict = proximity_vector_labels_dict
        self.k = k
        self.recall_scores = dict()

    def get_query_relevance(self, query, result):
        query = np.array(query)
        result = np.array(result)
        rel = query[query == result]
        if sum(rel) == 0  and len(rel) == self.n_classes:
            # query and result are in the "none" class i.e. their labels are all zeros
            # in this case, assign a relevance of 1.
            rel_score = 1.0
        elif len(rel) == 0:
            # no elements in common between query and result
            # relevance is zero in this case
            rel_score = 0.0
        else:
            rel_score = sum(rel)/max(sum(query), sum(result))
        return rel_score

    def relevance_metric(self, query_label, results_list):
        query_vector = self.proximity_vector_labels_dict[query_label]
        query_results_list = [self.proximity_vector_labels_dict[r] for r in results_list]
        query_relevances = [self.get_query_relevance(query_vector, result_vector) for result_vector in query_results_list]
        return query_relevances

    def calculate_recall(self, relevance_scores):
        return [sum(relevance_scores[:k])/k for k in self.k]
            
        #recall_at_k = [sum(results_labels[:k] == query_label)/k for k in self.k]
        #return recall_at_k

    def __call__(self, outputs, target, loss, n_triplets_list):
        distances_matrix = pdist(outputs)
        sorted_args = np.argsort(distances_matrix)
        sorted_args = sorted_args[:, 1:] # remove first column because it is the self-distance (always zero)
        recall_scores = []
        for i, row in enumerate(sorted_args):
            row = row.cpu().numpy()
            current_sample_label = target[i].item()
            row_labels = [target[i].item() for i in row]
            relevance_scores = self.relevance_metric(current_sample_label, row_labels)
            recall_scores.append(self.calculate_recall(relevance_scores))
        self.recall_scores = np.array(recall_scores)
        return self.value()
    

    def reset(self):
        self.recall_scores = dict()

    def value(self):
        return {k: round(np.mean(self.recall_scores[:, i]), 4) for i, k in enumerate(self.k)}

    def name(self):
        k_str_list = [f'k={k}' for k in self.k]
        return 'Recall@K: '+'; '.join(k_str_list)

'''


class Recall(Metric):

    #Calculates the Recall in a set of items.
    #Each item is interpreted as a query against the rest of the set.
    #Matches are calculated from each item's labels
    
    def __init__(self, proximity_vector_labels_dict, classes_list, k=[1, 5, 10]):
        self.proximity_vector_labels_dict = proximity_vector_labels_dict
        self.classes_list = classes_list
        self.n_classes = len(self.classes_list)
        self.relevance_orders = self.get_label_relevance_orders()
        self.k = k
        self.rng = np.random.default_rng(seed=0)
        self.metric_value = None
        self.recall_scores = []
        self.per_class_recall_scores = []
        self.recall_scores_random = []
        self.precision_scores = []
        self.per_class_precision_scores = []
        self.precision_scores_random = []
        self.mean_average_precision_scores = []
        self.per_class_mean_average_precision_scores = []
        self.mean_average_precision_scores_random = []
        self.precision_per_class = dict()
        self.average_precision_per_class = dict()
        self.recall_per_class = dict()
        self.recall_aggregated = dict()
        self.precision_aggregated = dict()
        self.average_precision_aggregated = dict()
    
    def get_query_per_class_relevance(self, query, result):
        query = np.array(query)
        result = np.array(result)
        rel = (query & result).astype(int)
        return rel

    def get_query_relevance(self, query, result):
        query = np.array(query)
        result = np.array(result)
        # rel = query[query == result]
        # if sum(rel) == 0  and len(rel) == self.n_classes:
        #     # query and result are in the "none" class i.e. their labels are all zeros
        #     # in this case, assign a relevance of 1.
        #     rel_score = 1.0
        # elif len(rel) == 0:
        #     # no elements in common between query and result
        #     # relevance is zero in this case
        #     rel_score = 0.0
        # else:
        #     rel_score = sum(rel)/max(sum(query), sum(result))
        return jaccard_score(query, result, zero_division=1.0)

    def get_label_relevance_orders(self):
        inter_label_relevance_dict = dict()
        for label_id, label_vector in self.proximity_vector_labels_dict.items():
            inter_label_relevance_dict[label_id] = [
                self.get_query_relevance(
                    label_vector, 
                    self.proximity_vector_labels_dict[r]
                ) for r in self.proximity_vector_labels_dict
             ]
        ordered_relevance_indices_dict = {
            label: list(np.flip(np.argsort(inter_label_relevance_dict[label]))) for label in inter_label_relevance_dict
        }
        return ordered_relevance_indices_dict

    def per_class_relevance_metric(self, query_vector, query_results_vector_list):
        query_relevances = [self.get_query_per_class_relevance(query_vector, result_vector) for result_vector in query_results_vector_list]
        return query_relevances

    def relevance_metric(self, query_vector, query_results_list):
        query_relevances = [self.get_query_relevance(query_vector, result_vector) for result_vector in query_results_list]
        return query_relevances


    def calculate_per_class_precision(self, query_vector, relevances):
        relevances = np.array(relevances)
        k_top = [min(k, len(relevances)) for k in self.k]
        per_class_precision = dict()
        for i, class_is_queried in enumerate(query_vector):
            if class_is_queried == 0:
                continue
            current_class = self.classes_list[i]
            per_class_precision[current_class] = {k: np.sum(relevances[:k, i])/k for k in k_top}
        return per_class_precision
    
    def calculate_precision(self, relevances):
        relevances = np.array(relevances)
        k_top = [min(k, len(relevances)) for k in self.k]
        per_class_precision = {k_original: np.sum(relevances[:k])/k for k, k_original in zip(k_top, self.k)}
        return per_class_precision
    
    def calculate_per_class_average_precision(self, query_vector, relevances):
        relevances = np.array(relevances)
        k_top = [min(k, len(relevances)) for k in self.k]
        per_class_avg_precision = dict()
        for i, class_is_queried in enumerate(query_vector):
            if class_is_queried == 0:
                continue
            current_class = self.classes_list[i]
            per_class_avg_precision[current_class] = {k: np.sum([relevances[j, i] * np.sum(relevances[:j, i])/(j+1) for j in range(k)])/k for k in k_top}
        return per_class_avg_precision
    
    def calculate_average_precision(self, relevances):
        relevances = np.array(relevances)
        k_top = [min(k, len(relevances)) for k in self.k]
        avg_precision = {k_original: np.sum([relevances[i] * np.sum(relevances[:i])/(i+1) for i in range(k)])/k for k, k_original in zip(k_top, self.k)}
        return avg_precision

    def calculate_per_class_recall(self, query_vector, relevances):
        relevances = np.array(relevances)
        k_top = [min(k, len(relevances)) for k in self.k]
        total_relevants_per_class = np.sum(relevances, axis=0)
        per_class_recall = dict()
        for i, class_is_queried in enumerate(query_vector):
            if class_is_queried == 0:
                continue
            current_class = self.classes_list[i]
            per_class_recall[current_class] = {k: np.sum(relevances[:k, i])/max(1, total_relevants_per_class[i]) for k in k_top}
        return per_class_recall
    
    def calculate_recall(self, relevances):
        total_relevants = np.sum(relevances, axis=-1)
        relevances = np.array(relevances)
        k_top = [min(k, len(relevances)) for k in self.k]
        recall = {k_original: np.sum(relevances[:k])/max(1, total_relevants) for k, k_original in zip(k_top, self.k)}
        return recall

    def __call__(self, epoch_number, dataset_embeddings, dataset_labels, query_embeddings, query_labels, loss, n_triplets_list, tensorboard_writer, training=True):
        print('Call on Metric.Recall')
        distances_matrix = query_dataset_dist(query_embeddings, dataset_embeddings)
        sorted_distance_matrix, sorted_args = distances_matrix.sort(axis=-1) # sorted distances in increasing order
        if training:
            sorted_distance_matrix = sorted_distance_matrix[:, 1:].numpy() 
            sorted_args = sorted_args[:, 1:].numpy() # remove first column because it is the self-distance (always zero)
        dataset_labels = dataset_labels.numpy()
        query_labels = query_labels.numpy()
        query_vectors = np.array([self.proximity_vector_labels_dict[q.item()] for q in query_labels])
        first_results_args = sorted_args[:, 0]
        first_results_labels = np.array([dataset_labels[r] for r in first_results_args])
        first_results_vectors = np.array([self.proximity_vector_labels_dict[r.item()] for r in first_results_labels])
        '''
        results_label_vectors = np.array([[self.proximity_vector_labels_dict[target[i]] for i in row] for row in sorted_args])
        ideal_results_args = np.array([sorted(row, key=lambda v: self.relevance_orders[target[i]].index(target[v])) for i, row in enumerate(sorted_args)])
        ideal_results_label_vectors = np.array([[self.proximity_vector_labels_dict[target[i]] for i in row] for row in ideal_results_args])
        self.ndcg_scores = [ndcg_score(ideal_result, result, k=self.k) for ideal_result, result in zip(ideal_results_label_vectors, results_label_vectors)]
        return self.value()
        '''
        for i, row in enumerate(sorted_args):
            #row = row.cpu().numpy().tolist()
            current_query_vector = query_vectors[i]
            results_labels = [dataset_labels[i].item() for i in row]
            results_vector_list = [self.proximity_vector_labels_dict[r] for r in results_labels]
            per_class_relevance_scores = self.per_class_relevance_metric(current_query_vector, results_vector_list)
            relevance_scores = self.relevance_metric(current_query_vector, results_vector_list)
            #relevance_scores = self.relevance_metric(current_query_label, results_labels)
            #relevance_scores.append(relevance_row)
            #results_ideal_scores_order = sorted(row, key=lambda v: self.relevance_orders[current_query_label].index(dataset_labels[v]))
            #results_ideal_scores_labels = [dataset_labels[i].item() for i in results_ideal_scores_order]
            print('Calculating recall_scores')
            self.per_class_recall_scores.append(self.calculate_per_class_recall(current_query_vector, per_class_relevance_scores))
            self.recall_scores.append(self.calculate_recall(relevance_scores))
            self.recall_scores_random.append(self.calculate_recall(self.rng.choice(relevance_scores, size=len(relevance_scores), replace=False)))
            print('Calculating precision_scores')
            self.per_class_precision_scores.append(self.calculate_per_class_precision(current_query_vector, per_class_relevance_scores))
            self.precision_scores.append(self.calculate_precision(relevance_scores))
            self.precision_scores_random.append(self.calculate_precision(self.rng.choice(relevance_scores, size=len(relevance_scores), replace=False)))
            print('Calculating average_precision_scores')
            self.per_class_mean_average_precision_scores.append(self.calculate_per_class_average_precision(current_query_vector, per_class_relevance_scores))
            self.mean_average_precision_scores.append(self.calculate_average_precision(relevance_scores))
            self.mean_average_precision_scores_random.append(self.calculate_average_precision(self.rng.choice(relevance_scores, size=len(relevance_scores), replace=False)))
            
            #self.ndcg_scores.append(self.calculate_ndcg(relevance_scores, ideal_relevance_scores))
            '''
            #results_vectors = np.array([self.proximity_vector_labels_dict[l] for l in row_labels])
            #ideal_results_vectors = np.array([self.proximity_vector_labels_dict[l] for l in row_ideal_scores_labels_order])
            #sklearn_ndcg_scores = {class_name: {k: [] for k in self.k} for class_name in self.classes_list}
            #for j, class_name in enumerate(self.classes_list):
            #    for k_value in self.k:
            #        sklearn_ndcg_scores[class_name][k_value].append(ndcg_score(ideal_results_vectors[:, j], results_vectors[:, j], k=k_value))
            #self.ndcg_scores = sklearn_ndcg_scores
            '''
            
        self.metric_value = self.value()
        if training:
            log_name_prefix = 'training'
        else:
            log_name_prefix = 'validation'
        for metric_name in self.metric_value:
            for k in self.metric_value[metric_name]:
                composed_metric_name = f'{log_name_prefix}_{metric_name}@{k}'
                tensorboard_writer.add_scalar(composed_metric_name, self.metric_value[metric_name][k], epoch_number)
    
        for i, class_name in enumerate(self.classes_list):
            y_true = query_vectors[:, i]
            y_pred = first_results_vectors[:, i]
            composed_metric_name = f'{log_name_prefix}_PR_curve_{class_name}'
            tensorboard_writer.add_pr_curve(composed_metric_name, y_true, y_pred)
        return self.metric_value
        
        
    

    def reset(self):
        self.metric_value = None
        self.recall_scores = []
        self.per_class_recall_scores = []
        self.recall_scores_random = []
        self.precision_scores = []
        self.per_class_precision_scores = []
        self.precision_scores_random = []
        self.mean_average_precision_scores = []
        self.per_class_mean_average_precision_scores = []
        self.mean_average_precision_scores_random = []
        self.precision_per_class = dict()
        self.average_precision_per_class = dict()
        self.recall_per_class = dict()
        self.recall_aggregated = dict()
        self.precision_aggregated = dict()
        self.average_precision_aggregated = dict()

    def value(self):
        print('Call on Metric.Recall.value()')
        # self.value() result is stored in the variable self.metric_value
        # If self.metric_value is already calculated, return it
        if self.metric_value is not None:
            return self.metric_value

        # TODO: Calculate metrics from randomized results
        # TODO: Calculate query-wise metrics

         # self.value() result is stored in the variable self.ndcg_value
        # If self.ndcg_value is already calculated, return it
        
        # calculate per query metrics & metrics on random
        print('Accumulating per query and randomized recall scores')
        recall = {k: 0.0 for k in self.k}
        recall_random = {k: 0.0 for k in self.k}
        query_count = 0
        for recall_scores, recall_random_scores in zip(self.recall_scores, self.recall_scores_random):
            query_count += 1
            for k_value in recall_scores:
                recall[k_value] += recall_scores[k_value]
                recall_random[k_value] += recall_random_scores[k_value]
        for k_value in recall:
            recall[k_value] /= query_count
            recall[k_value] = round(recall[k_value], 4)
            recall_random[k_value] /= query_count
            recall_random[k_value] = round(recall_random[k_value], 4)
        self.recall = recall
        self.recall_random = recall_random

        print('Accumulating per query and randomized precision scores')
        precision = {k: 0.0 for k in self.k}
        precision_random = {k: 0.0 for k in self.k}
        query_count = 0
        for precision_scores, precision_random_scores in zip(self.precision_scores, self.precision_scores_random):
            query_count += 1
            for k_value in precision_scores:
                precision[k_value] += precision_scores[k_value]
                precision_random[k_value] += precision_random_scores[k_value]
        for k_value in precision:
            precision[k_value] /= query_count
            precision[k_value] = round(precision[k_value], 4)
            precision_random[k_value] /= query_count
            precision_random[k_value] = round(precision_random[k_value], 4)
        self.precision = precision
        self.precision_random = precision_random

        print('Accumulating per query and randomized mAP scores')
        mean_average_precision = {k: 0.0 for k in self.k}
        mean_average_precision_random = {k: 0.0 for k in self.k}
        query_count = 0
        for mean_average_precision_scores, mean_average_precision_random_scores in zip(self.mean_average_precision_scores, self.mean_average_precision_scores_random):
            query_count += 1
            for k_value in mean_average_precision_scores:
                mean_average_precision[k_value] += mean_average_precision_scores[k_value]
                mean_average_precision_random[k_value] += mean_average_precision_random_scores[k_value]
        for k_value in mean_average_precision:
            mean_average_precision[k_value] /= query_count
            mean_average_precision[k_value] = round(mean_average_precision[k_value], 4)
            mean_average_precision_random[k_value] /= query_count
            mean_average_precision_random[k_value] = round(mean_average_precision_random[k_value], 4)
        self.mean_average_precision = mean_average_precision
        self.mean_average_precision_random = mean_average_precision_random

        precision_class_query_count = dict()
        average_precision_class_query_count = dict()
        recall_class_query_count = dict()
        precision_per_class = dict()
        average_precision_per_class = dict()
        recall_per_class = dict()
        recall_aggregated = dict()
        precision_aggregated = dict()
        average_precision_aggregated = dict()
        total_queries = 0
        print('Metric.Recall.value(): Accumulating scores')
        for precision_score, average_precision_score, recall_score in zip(self.per_class_precision_scores, self.per_class_mean_average_precision_scores, self.per_class_recall_scores):
            for precision_class_name, average_precision_class_name, recall_class_name in zip(precision_score, average_precision_score, recall_score):
                if precision_class_name not in precision_per_class:
                    precision_per_class[precision_class_name] = {k: 0.0 for k in self.k}
                if average_precision_class_name not in average_precision_per_class:
                    average_precision_per_class[average_precision_class_name] = {k: 0.0 for k in self.k}
                if recall_class_name not in recall_per_class:
                    recall_per_class[recall_class_name] = {k: 0.0 for k in self.k}
                if precision_class_name not in precision_class_query_count:
                    precision_class_query_count[precision_class_name] = 0
                if average_precision_class_name not in average_precision_class_query_count:
                    average_precision_class_query_count[average_precision_class_name] = 0
                if recall_class_name not in recall_class_query_count:
                    recall_class_query_count[recall_class_name] = 0
                precision_class_query_count[precision_class_name] += 1
                average_precision_class_query_count[average_precision_class_name] += 1
                recall_class_query_count[recall_class_name] += 1
                total_queries += 1
                for precision_k_value, average_precision_k_value, recall_k_value in zip(precision_score[precision_class_name], average_precision_score[average_precision_class_name], recall_score[recall_class_name]):
                    precision_per_class[precision_class_name][precision_k_value] += precision_score[precision_class_name][precision_k_value]
                    average_precision_per_class[average_precision_class_name][average_precision_k_value] += average_precision_score[average_precision_class_name][average_precision_k_value]
                    recall_per_class[recall_class_name][recall_k_value] += recall_score[recall_class_name][recall_k_value]
        print('Metric.Recall.value(): Averaging scores')
        for precision_class_name, average_precision_class_name, recall_class_name in zip(precision_per_class, average_precision_per_class, recall_per_class):
            for precision_k_value, average_precision_k_value, recall_k_value in zip(precision_per_class[precision_class_name], average_precision_per_class[average_precision_class_name], recall_per_class[recall_class_name]):
                
                precision_per_class[precision_class_name][precision_k_value] /= precision_class_query_count[precision_class_name]
                if precision_k_value not in precision_aggregated:
                    precision_aggregated[precision_k_value] = 0.0
                precision_aggregated[precision_k_value] += precision_per_class[precision_class_name][precision_k_value] * (precision_class_query_count[precision_class_name]/total_queries)
                precision_per_class[precision_class_name][precision_k_value] = round(precision_per_class[precision_class_name][precision_k_value], 4)
                
                average_precision_per_class[average_precision_class_name][average_precision_k_value] /= average_precision_class_query_count[average_precision_class_name]
                if average_precision_k_value not in average_precision_aggregated:
                    average_precision_aggregated[average_precision_k_value] = 0.0
                average_precision_aggregated[average_precision_k_value] += average_precision_per_class[average_precision_class_name][average_precision_k_value] * (average_precision_class_query_count[average_precision_class_name]/total_queries)
                average_precision_per_class[average_precision_class_name][average_precision_k_value] = round(average_precision_per_class[average_precision_class_name][average_precision_k_value], 4)
                
                recall_per_class[recall_class_name][recall_k_value] /= recall_class_query_count[recall_class_name]
                if recall_k_value not in recall_aggregated:
                    recall_aggregated[recall_k_value] = 0.0
                recall_aggregated[recall_k_value] += recall_per_class[recall_class_name][recall_k_value] * (recall_class_query_count[recall_class_name]/total_queries)
                recall_per_class[recall_class_name][recall_k_value] = round(recall_per_class[recall_class_name][recall_k_value], 4)
        
        precision_aggregated = {k: round(v, 4) for k, v in precision_aggregated.items()}
        average_precision_aggregated = {k: round(v, 4) for k, v in average_precision_aggregated.items()}
        recall_aggregated = {k: round(v, 4) for k, v in recall_aggregated.items()}
        self.precision_per_class = precision_per_class
        self.average_precision_per_class = average_precision_per_class
        self.recall_per_class = recall_per_class
        self.precision_aggregated = precision_aggregated
        self.average_precision_aggregated = average_precision_aggregated
        self.recall_aggregated = recall_aggregated
        print('Metric.Recall.value(): Finished. Now returning...')
        return dict([('precision', self.precision)]+
                    [('precision_random', self.precision_random)]+
                    [('precision_aggregated', self.precision_aggregated)]+
                    [(f'precision_{cname}', self.precision_per_class[cname]) for cname in self.precision_per_class]+
                    [('mean_average_precision', self.mean_average_precision)]+
                    [('mean_average_precision_random', self.mean_average_precision_random)]+
                    [('mean_average_precision_aggregated', self.average_precision_aggregated)]+
                    [(f'mean_average_precision_{cname}', self.average_precision_per_class[cname]) for cname in self.average_precision_per_class]+
                    [('recall', self.recall)]+
                    [('recall_random', self.recall_random)]+
                    [('recall_aggregated', self.recall_aggregated)]+
                    [(f'recall_{cname}', self.recall_per_class[cname]) for cname in self.recall_per_class])
    
        

    def name(self):
        return f'Precision@k, mAP@k & Recall@k (k={self.k})'
    
class AllMetrics(Metric):

    #Calculates Recall, Precision, mAP & NDCG for a set of queries and their results.
    #Each item is interpreted as a query against the rest of the set.
    #Matches are calculated from each item's labels
    
    def __init__(self, proximity_vector_labels_dict, classes_list, k=[1, 5, 10]):
        self.proximity_vector_labels_dict = proximity_vector_labels_dict
        self.classes_list = classes_list
        self.n_classes = len(self.classes_list)
        self.relevance_orders = self.get_label_relevance_orders()
        self.k = k
        self.rng = np.random.default_rng(seed=0)
        self.metric_value = None
        self.recall_scores = []
        self.per_class_recall_scores = []
        self.recall_scores_random = []
        self.per_class_recall_scores_random = []
        self.per_class_precision_scores_random = []
        self.f1_scores= []
        self.f1_scores_random = []
        self.per_class_f1_scores = []
        self.per_class_f1_scores_random = []
        self.macro_f1_scores = []
        self.macro_f1_scores_random = []
        self.precision_scores = []
        self.per_class_precision_scores = []
        self.precision_scores_random = []
        self.mean_average_precision_scores = []
        self.per_class_mean_average_precision_scores = []
        self.mean_average_precision_scores_random = []
        self.ndcg_scores = []
        self.ndcg_random = [] 
        self.per_class_ndcg_scores = []
        self.ndcg_per_class = dict()
        self.ndcg_aggregated = dict()
        self.precision_per_class = dict()
        self.average_precision_per_class = dict()
        self.recall_per_class = dict()
        self.recall_aggregated = dict()
        self.precision_aggregated = dict()
        self.average_precision_aggregated = dict()
        self.f1_scores = dict()
        self.label_vector_helper = LabelVectorHelper()

        self.log_interval = 1
        self.train_call_counter = 0
        self.test_call_counter = 0
    
    def get_query_per_class_relevance(self, query, result):
        query = np.array(query)
        result = np.array(result)
        rel = (query & result).astype(int)
        return rel

    def get_query_relevance(self, query, result):
        query = np.array(query)
        result = np.array(result)
        # rel = query[query == result]
        # if sum(rel) == 0  and len(rel) == self.n_classes:
        #     # query and result are in the "none" class i.e. their labels are all zeros
        #     # in this case, assign a relevance of 1.
        #     rel_score = 1.0
        # elif len(rel) == 0:
        #     # no elements in common between query and result
        #     # relevance is zero in this case
        #     rel_score = 0.0
        # else:
        #     rel_score = sum(rel)/max(sum(query), sum(result))
        return jaccard_score(query, result, zero_division=1.0)

    def get_label_relevance_orders(self):
        inter_label_relevance_dict = dict()
        for label_id, label_vector in self.proximity_vector_labels_dict.items():
            inter_label_relevance_dict[label_id] = [
                self.get_query_relevance(
                    label_vector, 
                    self.proximity_vector_labels_dict[r]
                ) for r in self.proximity_vector_labels_dict
             ]
        ordered_relevance_indices_dict = {
            label: list(np.flip(np.argsort(inter_label_relevance_dict[label]))) for label in inter_label_relevance_dict
        }
        return ordered_relevance_indices_dict

    def per_class_relevance_metric(self, query_vector, query_results_vector_list):
        query = np.array(query_vector) # query vector
        results = np.array(query_results_vector_list) # lists of results as vectors
        query_relevances = (query & results).astype(int)
        return query_relevances

    def relevance_metric(self, query_vector, query_results_list):
        query = np.array(query_vector) # query vector
        results = np.array(query_results_list) # lists of results as vectors
        query_relevances = np.sum(query & results, axis=-1) / np.sum(query | results, axis=-1) # Jaccard similarity
        return query_relevances


    def calculate_per_class_precision(self, query_vector, relevances):
        relevances = np.array(relevances)
        k_top = [min(k, len(relevances)) for k in self.k]
        per_class_precision = dict()
        for i, class_is_queried in enumerate(query_vector):
            if class_is_queried == 0:
                continue
            current_class = self.classes_list[i]
            per_class_precision[current_class] = {k: np.sum(relevances[:k, i])/k for k in k_top}
        return per_class_precision
    
    def calculate_precision(self, relevances):
        relevances = np.array(relevances)
        k_top = [min(k, len(relevances)) for k in self.k]
        per_class_precision = {k_original: np.sum(relevances[:k])/k for k, k_original in zip(k_top, self.k)}
        return per_class_precision
    
    # def calculate_per_class_average_precision(self, query_vector, relevances):
    #     relevances = np.array(relevances)
    #     k_top = [min(k, len(relevances)) for k in self.k]
    #     per_class_avg_precision = dict()
    #     for i, class_is_queried in enumerate(query_vector):
    #         if class_is_queried == 0:
    #             continue
    #         current_class = self.classes_list[i]
    #         per_class_avg_precision[current_class] = {k: np.sum([relevances[j, i] * np.sum(relevances[:j+1, i])/(j+1) for j in range(k)])/k for k in k_top}
    #     return per_class_avg_precision
    
    def calculate_per_class_average_precision(self, query_vector, relevances):
        relevances = np.array(relevances)
        k_top = [min(k, len(relevances)) for k in self.k]
        per_class_avg_precision = {}

        for i, class_is_queried in enumerate(query_vector):
            if class_is_queried == 0:
                continue
            current_class = self.classes_list[i]
            per_class_avg_precision[current_class] = {}

            rel_column = relevances[:, i]

            for k_val, k_original in zip(k_top, self.k):
                rel_k = rel_column[:k_val]
                if rel_k.sum() == 0:
                    per_class_avg_precision[current_class][k_original] = 0.0
                    continue
                precisions = [(rel_k[:j+1].sum()) / (j+1) for j in range(k_val)]
                weighted_precisions = [p * r for p, r in zip(precisions, rel_k)]
                ap = np.sum(weighted_precisions) / rel_k.sum()
                per_class_avg_precision[current_class][k_original] = ap

        return per_class_avg_precision

    # def calculate_average_precision(self, relevances):
    #     relevances = np.array(relevances)
    #     k_top = [min(k, len(relevances)) for k in self.k]
    #     avg_precision = {k_original: np.sum([relevances[i] * np.sum(relevances[:i+1])/(i+1) for i in range(k)])/k for k, k_original in zip(k_top, self.k)}
    #     return avg_precision

    def calculate_average_precision(self, relevances):
        relevances = np.array(relevances)
        k_top = [min(k, len(relevances)) for k in self.k]
        avg_precision = {}

        for k_val, k_original in zip(k_top, self.k):
            rel_k = relevances[:k_val]
            rel_cumsum = np.cumsum(rel_k)
            precisions = rel_cumsum / np.arange(1, k_val + 1) # P(i)
            weighted_precisions = precisions * rel_k # P(i) * r_i

            denom = rel_k.sum()
            ap = weighted_precisions.sum() / denom if denom > 0 else 0.0
            avg_precision[k_original] = ap

        return avg_precision

    def calculate_per_class_recall(self, query_vector, relevances):
        relevances = np.array(relevances)
        per_class_recall = dict()
        for i, class_is_queried in enumerate(query_vector):
            if class_is_queried == 0:
                continue
            sorted_relevants = -np.sort(-1 * relevances[:, i])
            last_relevant_idx = np.where(sorted_relevants > 0)[0]
            if len(last_relevant_idx) > 0:
                last_relevant_idx = last_relevant_idx[-1]
            else:
                last_relevant_idx = 0
            n_relevants = last_relevant_idx + 1
            k_top = [min(k, len(relevances), n_relevants) for k in self.k]
            current_class = self.classes_list[i]
            per_class_recall[current_class] = {k: np.sum(relevances[:k_bounded, i])/max(1, np.sum(sorted_relevants[:k_bounded])) for k, k_bounded in zip(self.k, k_top)}
        return per_class_recall

    def calculate_recall(self, relevances):
        sorted_relevants = -np.sort(-1 * relevances)
        last_relevant_idx = np.where(sorted_relevants > 0)[0]
        if len(last_relevant_idx) > 0:
            last_relevant_idx = last_relevant_idx[-1]
        else:
            last_relevant_idx = 0
        n_relevants = last_relevant_idx + 1
        relevances = np.array(relevances)
        k_top = [min(k, len(relevances), n_relevants) for k in self.k]
        recall = {k_original: np.sum(relevances[:k])/max(1, np.sum(sorted_relevants[:k])) for k, k_original in zip(k_top, self.k)}
        return recall
    
    def calculate_f1(self, precision_at_k, recall_at_k):
        f1_at_k = dict()
        for k in self.k:
            p = precision_at_k[k]
            r = recall_at_k[k]
            f1_at_k[k] = (2 * p * r) / (p + r + 1e-7)
        return f1_at_k

    def calculate_per_class_f1(self, per_class_precision_at_k, per_class_recall_at_k):
        per_class_f1_at_k = dict()
        for c in per_class_precision_at_k:
            per_class_f1_at_k[c] = dict()
            for k in per_class_precision_at_k[c]:
                p = per_class_precision_at_k[c][k]
                r = per_class_recall_at_k[c][k]
                per_class_f1_at_k[c][k] = (2 * p * r) / (p + r + 1e-7)
        return per_class_f1_at_k

    def calculate_macro_f1(self, per_class_f1_at_k):
        macro_f1_at_k = dict()
        c_list = list(per_class_f1_at_k.keys())
        if not c_list:  # Return empty dict if no classes
            return macro_f1_at_k
        k_list = list(per_class_f1_at_k[c_list[0]].keys())  # Use keys(), not values()
        for k in k_list:
            f1_values_list = []
            for c in c_list:
                f1_values_list.append(per_class_f1_at_k[c][k])
            macro_f1_at_k[k] = np.mean(f1_values_list)  # Use k as key, not c
        return macro_f1_at_k
    
    def calculate_per_class_ndcg(self, query_label, relevances, ideal_relevances):
        k_top = [min(k, len(relevances)) for k in self.k]
        query_vector = self.proximity_vector_labels_dict[query_label]
        dcg_at_k = np.array([relevances[i]/np.log2((i+1)+1) for i in range(max(k_top))])
        ideal_dcg_at_k = np.array([ideal_relevances[i]/np.log2((i+1)+1) for i in range(max(k_top))])
        per_class_ndcg = dict()
        for i, class_is_queried in enumerate(query_vector):
            if class_is_queried == 0:
                continue
            # current_class := class name
            current_class = self.classes_list[i] 
            per_class_ndcg[current_class] = {k_original: np.sum(dcg_at_k[:k, i])/max(1., np.sum(ideal_dcg_at_k[:k, i])) for k, k_original in zip(k_top, self.k)}
        return per_class_ndcg

    def calculate_ndcg(self, relevances, ideal_relevances):
        k_top = [min(k, len(relevances)) for k in self.k]
        dcg_at_k = np.array([relevances[i]/np.log2((i+1)+1) for i in range(max(k_top))])
        ideal_dcg_at_k = np.array([ideal_relevances[i]/np.log2((i+1)+1) for i in range(max(k_top))])
        ndcg_at_k = {k_original: np.sum(dcg_at_k[:k])/max(1., np.sum(ideal_dcg_at_k[:k])) for k, k_original in zip(k_top, self.k)}
        return ndcg_at_k
    
    def compute_f1_scores(self, preds: np.ndarray, targets: np.ndarray):
        """
        Args:
            preds: binary prediction matrix of shape [N, C]
            targets: binary ground truth matrix of shape [N, C]

        Returns:
            dict with:
                - micro_f1
                - macro_f1
                - weighted_micro_f1
                - weighted_macro_f1
                - per_class_micro_f1: {class_name: score}
                - per_class_macro_f1: {class_name: score}
        """
        preds = (preds > 0).astype(int)
        targets = (targets > 0).astype(int)

        tp = (preds & targets).sum()
        fp = (preds & ~targets).sum()
        fn = (~preds & targets).sum()

        micro_precision = tp / max(tp + fp, 1)
        micro_recall = tp / max(tp + fn, 1)
        micro_f1 = 2 * micro_precision * micro_recall / max(micro_precision + micro_recall, 1e-8)

        tp_c = (preds & targets).sum(axis=0)
        fp_c = (preds & ~targets).sum(axis=0)
        fn_c = (~preds & targets).sum(axis=0)

        precision_c = tp_c / np.maximum(tp_c + fp_c, 1)
        recall_c = tp_c / np.maximum(tp_c + fn_c, 1)
        f1_c = 2 * precision_c * recall_c / np.maximum(precision_c + recall_c, 1e-8)

        support = tp_c + fn_c  # ground truth positives per class
        total_support = np.maximum(support.sum(), 1)

        weighted_f1_c = f1_c * support / total_support

        # Build per-class dicts
        per_class_micro_f1 = {cls: round(float(f1_c[i]), 4) for i, cls in enumerate(self.classes_list)}
        per_class_macro_f1 = {cls: round(float(f1_c[i]), 4) for i, cls in enumerate(self.classes_list)}

        return {
            "micro_f1": round(float(micro_f1), 4),
            "macro_f1": round(float(f1_c.mean()), 4),
            "weighted_micro_f1": round(float((f1_c * support).sum() / total_support), 4),
            "weighted_macro_f1": round(float((f1_c * support).sum() / total_support), 4),
            "per_class_micro_f1": per_class_micro_f1,
            "per_class_macro_f1": per_class_macro_f1
        }


    def __call__(self, epoch_number, dataset_embeddings, dataset_labels, query_embeddings, query_predicted_logits, query_labels, loss, n_triplets_list, tensorboard_writer, training=True):
        if training:
            mode_string = "[Training]"
            self.train_call_counter += 1
        else:
            mode_string = "[Validation]"
            self.test_call_counter += 1
        print(f'{mode_string} Call on Metric.AllMetrics')
        distances_matrix = query_dataset_dist(query_embeddings, dataset_embeddings) # <-- Squared Euclidean distance (0.0 means identical data points)
        #distances_matrix = query_dataset_sim_cosine(query_embeddings, dataset_embeddings) # <-- Cosine similarity!! (1.0 means identical data points)
        if self.train_call_counter % self.log_interval == 0 or self.test_call_counter % self.log_interval == 0:
            print('distances_matrix[0]:\n', distances_matrix[0])
            print()
            print('distances_matrix:\n', distances_matrix)
        sorted_distance_matrix, sorted_args = distances_matrix.sort(axis=-1) # sorted distances in increasing order for Squared Euclidean distance
        #sorted_distance_matrix, sorted_args = distances_matrix.sort(dim=-1, descending=True) # sorted distances in decreasing order for Cosine similarity

        if training:
            sorted_distance_matrix = sorted_distance_matrix[:, 1:].numpy() 
            sorted_args = sorted_args[:, 1:].numpy() # remove first column because it is the self-distance (always zero) 
        dataset_labels = dataset_labels.numpy()
        if not hasattr(dataset_labels[0], 'shape'):
            dataset_vectors = np.array([self.proximity_vector_labels_dict[d.item()] for d in dataset_labels])
        else:
            dataset_vectors = dataset_labels
            dataset_labels = [self.label_vector_helper.get_class_id(lbl.tolist()) for lbl in dataset_labels]
        query_labels = query_labels.numpy()
        if not hasattr(query_labels[0], 'shape'):
            query_vectors = np.array([self.proximity_vector_labels_dict[q.item()] for q in query_labels])
        else:
            query_vectors = query_labels
            query_labels = [self.label_vector_helper.get_class_id(lbl.tolist()) for lbl in query_labels]
        #print('query_predicted_logits.shape:', query_predicted_logits.shape)
        #print('query_labels.shape:', query_labels.shape)
        #print('query_predicted_logits:\n', query_predicted_logits)
        #print('query_labels:\n', query_labels)
        #sorted_results_args = sorted_args[:, 0]
        #sorted_results_labels = np.array([dataset_labels[r] for r in sorted_results_args])
        #if not hasattr(sorted_results_labels[0], 'shape'):
        #    sorted_results_vectors = np.array([self.proximity_vector_labels_dict[int(r.item())] for r in sorted_results_labels])
        #else:
        #    sorted_results_vectors = np.array([self.proximity_vector_labels_dict[int(r)] for r in sorted_results_labels])
        for i, row in enumerate(sorted_args):
            #row = row.cpu().numpy().tolist()
            #print(f'{mode_string} Processing query {i+1}/{len(sorted_args)}')
            #print(f'{mode_string} Getting current_query_label...')
            current_query_label = query_labels[i]
            #print(f'{mode_string} Getting current_query_vector...')
            current_query_vector = query_vectors[i]
            #print(f'{mode_string} Getting results_labels...')
            results_labels = [dataset_labels[i] for i in row]
            #print(f'{mode_string} Getting results_vector_list...')
            results_vector_list = [self.proximity_vector_labels_dict[r] for r in results_labels]
            results_vector_list_random = self.rng.choice(results_vector_list, size=len(results_vector_list), replace=False)
            #print(f'{mode_string} Getting per_class_relevance_scores...')
            per_class_relevance_scores = self.per_class_relevance_metric(current_query_vector, results_vector_list)
            per_class_relevance_scores_random = self.per_class_relevance_metric(current_query_vector, results_vector_list_random)
            #print(f'{mode_string} Getting relevance_scores...')
            relevance_scores = self.relevance_metric(current_query_vector, results_vector_list)
            #print(f'{mode_string} Getting relevance_scores_random...')
            relevance_scores_random = self.relevance_metric(current_query_vector, results_vector_list_random)
            #print(f'{mode_string} Getting results_ideal_scores_order...')
            results_ideal_scores_order = sorted(row, key=lambda v: self.relevance_orders[current_query_label].index(dataset_labels[v]))
            #print(f'{mode_string} Getting results_ideal_scores_labels...')
            results_ideal_scores_labels = [dataset_labels[i] for i in results_ideal_scores_order]
            #print(f'{mode_string} Getting results_ideal_scores_vectors...')
            results_ideal_scores_vectors = [self.proximity_vector_labels_dict[r] for r in results_ideal_scores_labels]
            #print(f'{mode_string} Getting ideal_relevance_scores...')
            ideal_relevance_scores = self.relevance_metric(current_query_vector, results_ideal_scores_vectors)
            #print(f'{mode_string} Getting per_class_ideal_relevance_scores...')
            per_class_ideal_relevance_scores = self.per_class_relevance_metric(current_query_vector, results_ideal_scores_vectors)
            #print(f'{mode_string} Calculating recall_scores')
            per_class_recall = self.calculate_per_class_recall(current_query_vector, per_class_relevance_scores)
            per_class_recall_random = self.calculate_per_class_recall(current_query_vector, per_class_relevance_scores_random)
            recall = self.calculate_recall(relevance_scores)
            recall_random = self.calculate_recall(relevance_scores_random)
            
            self.per_class_recall_scores.append(per_class_recall)
            self.per_class_recall_scores_random.append(per_class_recall_random)
            self.recall_scores.append(recall)
            self.recall_scores_random.append(recall_random)
            
            #print(f'{mode_string} Calculating precision_scores')
            per_class_precision = self.calculate_per_class_precision(current_query_vector, per_class_relevance_scores)
            per_class_precision_random = self.calculate_per_class_precision(current_query_vector, per_class_relevance_scores_random)
            precision = self.calculate_precision(relevance_scores)
            precision_random = self.calculate_precision(relevance_scores_random)
            
            self.per_class_precision_scores.append(per_class_precision)
            self.per_class_precision_scores_random.append(per_class_precision_random)
            self.precision_scores.append(precision)
            self.precision_scores_random.append(precision_random)
            
            #print(f'{mode_string} Calculating micro_f1_scores')
            # IMPORTANT: Micro F1 scores MUST be calculated after precision and recall in this scope!
            per_class_f1 = self.calculate_per_class_f1(per_class_precision, per_class_recall)
            per_class_f1_random = self.calculate_per_class_f1(per_class_precision_random, per_class_recall_random)
            f1 = self.calculate_f1(precision, recall)
            f1_random = self.calculate_f1(precision_random, recall_random)
            
            self.per_class_f1_scores.append(per_class_f1)
            self.per_class_f1_scores_random.append(per_class_f1_random)
            self.f1_scores.append(f1)
            self.f1_scores_random.append(f1_random)
            
            #print(f'{mode_string} Calculating macro_f1_scores')
            # IMPORTANT: Macro F1 scores MUST be calculated after Micro F1 scores in this scope!
            macro_f1 = self.calculate_macro_f1(per_class_f1)
            macro_f1_random = self.calculate_macro_f1(per_class_f1_random)
            
            self.macro_f1_scores.append(macro_f1)
            self.macro_f1_scores_random.append(macro_f1_random)
            #print(f'{mode_string} Calculating average_precision_scores')
            self.per_class_mean_average_precision_scores.append(self.calculate_per_class_average_precision(current_query_vector, per_class_relevance_scores))
            self.mean_average_precision_scores.append(self.calculate_average_precision(relevance_scores))
            self.mean_average_precision_scores_random.append(self.calculate_average_precision(relevance_scores_random))
            #print(f'{mode_string} Calculating ndcg_at_k')
            self.ndcg_scores.append(self.calculate_ndcg(relevance_scores, ideal_relevance_scores))
            #print(f'{mode_string} Calculating ndcg_at_k_random')
            self.ndcg_random.append(self.calculate_ndcg(relevance_scores_random, ideal_relevance_scores))
            #print(f'{mode_string} Calculating per_class_ndcg_scores')
            self.per_class_ndcg_scores.append(self.calculate_per_class_ndcg(current_query_label, per_class_relevance_scores, per_class_ideal_relevance_scores))  
        self.metric_value = self.value(training=training)
        if training:
            log_name_prefix = 'training'
        else:
            log_name_prefix = 'validation'
        for metric_name in self.metric_value:
            metric_data = self.metric_value[metric_name]
            
            # Check if metric_data is a dictionary (contains @k values)
            if isinstance(metric_data, dict) and metric_data:
                # Dictionary case: metric@k format
                for k, value in metric_data.items():
                    composed_metric_name = f'{log_name_prefix}_{metric_name}@{k}'
                    # Ensure the value is a valid scalar
                    scalar_value = float(value) if hasattr(value, '__float__') else value
                    tensorboard_writer.add_scalar(composed_metric_name, scalar_value, epoch_number)
            else:
                # Scalar case: just metric name
                composed_metric_name = f'{log_name_prefix}_{metric_name}'
                # Ensure the value is a valid scalar
                scalar_value = float(metric_data) if hasattr(metric_data, '__float__') else metric_data
                tensorboard_writer.add_scalar(composed_metric_name, scalar_value, epoch_number)

        # for i, class_name in enumerate(self.classes_list):
        #     y_true = query_vectors[:, i]
        #     y_pred = first_results_vectors[:, i]
        #     composed_metric_name = f'{log_name_prefix}_PR_curve_{class_name}'
        #     tensorboard_writer.add_pr_curve(composed_metric_name, y_true, y_pred)
        #self.f1_scores = self.compute_f1_scores(preds=np.array(query_predicted_logits), targets=np.array(query_vectors))

        # Log F1 scores to TensorBoard
        # for f1_metric_name, f1_value in self.f1_scores.items():
        #     if isinstance(f1_value, dict):
        #         # Per-class F1 scores
        #         for class_name, class_f1_value in f1_value.items():
        #             composed_metric_name = f'{log_name_prefix}_{f1_metric_name}_{class_name}'
        #             tensorboard_writer.add_scalar(composed_metric_name, class_f1_value, epoch_number)
        #     else:
        #         # Scalar F1 scores (micro_f1, macro_f1, etc.)
        #         composed_metric_name = f'{log_name_prefix}_{f1_metric_name}'
        #         tensorboard_writer.add_scalar(composed_metric_name, f1_value, epoch_number)
        
        return self.metric_value

    def reset(self):

        self.metric_value = None

        self.per_class_recall_scores_random = []
        self.per_class_precision_scores_random = []
        self.f1_scores = []
        self.f1_scores_random = []
        self.per_class_f1_scores_random = []
        self.per_class_f1_scores = []
        self.macro_f1_scores = []
        self.macro_f1_scores_random = []

        self.recall_scores = []
        self.precision_scores = []
        self.mean_average_precision_scores = []
        self.ndcg_scores = []
        
        self.per_class_precision_scores = []
        self.per_class_recall_scores = []
        self.per_class_mean_average_precision_scores = []
        self.per_class_ndcg_scores = []
        
        self.mean_average_precision_scores_random = []
        self.precision_scores_random = []
        self.recall_scores_random = []
        self.ndcg_random = []

        self.precision_per_class = dict()
        self.average_precision_per_class = dict()
        self.recall_per_class = dict()
        self.ndcg_per_class = dict()

        self.recall_aggregated = dict()
        self.precision_aggregated = dict()
        self.average_precision_aggregated = dict()
        self.ndcg_aggregated = dict()
        self.f1_aggregated = dict()
        
        # Random per-class aggregated metrics
        self.precision_aggregated_random = dict()
        self.recall_aggregated_random = dict()
        self.f1_aggregated_random = dict()
        
        # Per-class random metrics
        self.precision_per_class_random = dict()
        self.recall_per_class_random = dict()
        self.f1_per_class = dict()
        self.f1_per_class_random = dict()


    def _average_query_metrics(self, metric_scores, metric_scores_random):
        """Average metrics across all queries for aggregate metrics (non-class-specific)."""
        metric = {k: 0.0 for k in self.k}
        metric_random = {k: 0.0 for k in self.k}
        
        query_count = len(metric_scores)
        for scores, scores_random in zip(metric_scores, metric_scores_random):
            for k_value in scores:
                metric[k_value] += scores[k_value]
                metric_random[k_value] += scores_random[k_value]
        
        # Average and round
        for k_value in metric:
            metric[k_value] = float(round(metric[k_value] / query_count, 4))
            metric_random[k_value] = float(round(metric_random[k_value] / query_count, 4))
            
        return metric, metric_random

    def _process_per_class_metrics(self, per_class_metric_scores):
        """Process per-class metrics to get per-class averages and aggregated metrics."""
        # Initialize containers
        metric_per_class = {}
        metric_class_query_count = {}
        
        # Accumulate scores for each class
        for query_metrics in per_class_metric_scores:
            for class_name, class_metrics in query_metrics.items():
                if class_name not in metric_per_class:
                    metric_per_class[class_name] = {k: 0.0 for k in self.k}
                    metric_class_query_count[class_name] = 0
                
                metric_class_query_count[class_name] += 1
                for k_value, score in class_metrics.items():
                    metric_per_class[class_name][k_value] += score
        
        # Average per-class scores
        for class_name in metric_per_class:
            for k_value in metric_per_class[class_name]:
                metric_per_class[class_name][k_value] /= metric_class_query_count[class_name]
                metric_per_class[class_name][k_value] = float(round(metric_per_class[class_name][k_value], 4))
        
        # Calculate aggregated metrics as simple average of per-class averages
        metric_aggregated = {k: 0.0 for k in self.k}
        if metric_per_class:  # Only if we have per-class data
            num_classes = len(metric_per_class)
            for class_name in metric_per_class:
                for k_value in metric_per_class[class_name]:
                    metric_aggregated[k_value] += metric_per_class[class_name][k_value]
            
            # Average across classes
            for k_value in metric_aggregated:
                metric_aggregated[k_value] = float(round(metric_aggregated[k_value] / num_classes, 4))
        
        return metric_per_class, metric_aggregated

    def _build_result_dict(self):
        """Build the final result dictionary with all metrics."""
        result = {}
        
        # Add aggregate metrics (non-class-specific)
        result.update([
            ('mean_average_precision', self.mean_average_precision),
            ('NDCG', self.ndcg),
            ('precision', self.precision),
            ('recall', self.recall),
            ('f1', self.f1),
            ('macro_f1', self.macro_f1),
        ])
        
        # Add random baseline metrics
        result.update([
            ('mean_average_precision_random', self.mean_average_precision_random),
            ('NDCG_random', self.ndcg_random),
            ('precision_random', self.precision_random),
            ('recall_random', self.recall_random),
            ('f1_random', self.f1_random),
            ('macro_f1_random', self.macro_f1_random),
        ])
        
        # Add aggregated per-class metrics
        result.update([
            ('mean_average_precision_aggregated', self.average_precision_aggregated),
            ('NDCG_per_class_aggregated', self.ndcg_aggregated),
            ('precision_aggregated', self.precision_aggregated),
            ('recall_aggregated', self.recall_aggregated),
            ('f1_aggregated', self.f1_aggregated),
            ('precision_aggregated_random', self.precision_aggregated_random),
            ('recall_aggregated_random', self.recall_aggregated_random),
            ('f1_aggregated_random', self.f1_aggregated_random),
        ])
        
        # Add individual per-class metrics
        for class_name in self.average_precision_per_class:
            result[f'mean_average_precision_{class_name}'] = self.average_precision_per_class[class_name]
        for class_name in self.ndcg_per_class:
            result[f'NDCG_{class_name}'] = self.ndcg_per_class[class_name]
        for class_name in self.precision_per_class:
            result[f'precision_{class_name}'] = self.precision_per_class[class_name]
        for class_name in self.recall_per_class:
            result[f'recall_{class_name}'] = self.recall_per_class[class_name]
        for class_name in self.f1_per_class:
            result[f'f1_{class_name}'] = self.f1_per_class[class_name]
            
        # Add individual per-class random metrics
        for class_name in self.precision_per_class_random:
            result[f'precision_{class_name}_random'] = self.precision_per_class_random[class_name]
        for class_name in self.recall_per_class_random:
            result[f'recall_{class_name}_random'] = self.recall_per_class_random[class_name]
        for class_name in self.f1_per_class_random:
            result[f'f1_{class_name}_random'] = self.f1_per_class_random[class_name]
        
        return result

    def value(self, training=True):
        """Calculate and return all metrics averages."""
        mode_string = "[Training]" if training else "[Validation]"
        print(f'{mode_string} Call on Metric.AllMetrics.value()')
        
        # Return cached result if already calculated
        if self.metric_value is not None:
            return self.metric_value

        # Calculate aggregate metrics (averaged across all queries)
        self.precision, self.precision_random = self._average_query_metrics(
            self.precision_scores, self.precision_scores_random
        )
        self.recall, self.recall_random = self._average_query_metrics(
            self.recall_scores, self.recall_scores_random
        )
        self.mean_average_precision, self.mean_average_precision_random = self._average_query_metrics(
            self.mean_average_precision_scores, self.mean_average_precision_scores_random
        )
        self.ndcg, self.ndcg_random = self._average_query_metrics(
            self.ndcg_scores, self.ndcg_random
        )
        self.f1, self.f1_random = self._average_query_metrics(
            self.f1_scores, self.f1_scores_random
        )
        self.macro_f1, self.macro_f1_random = self._average_query_metrics(
            self.macro_f1_scores, self.macro_f1_scores_random
        )

        # Calculate per-class metrics and their aggregations
        self.precision_per_class, self.precision_aggregated = self._process_per_class_metrics(
            self.per_class_precision_scores
        )
        self.precision_per_class_random, self.precision_aggregated_random = self._process_per_class_metrics(
            self.per_class_precision_scores_random
        )
        self.recall_per_class, self.recall_aggregated = self._process_per_class_metrics(
            self.per_class_recall_scores
        )
        self.recall_per_class_random, self.recall_aggregated_random = self._process_per_class_metrics(
            self.per_class_recall_scores_random
        )
        self.average_precision_per_class, self.average_precision_aggregated = self._process_per_class_metrics(
            self.per_class_mean_average_precision_scores
        )
        self.ndcg_per_class, self.ndcg_aggregated = self._process_per_class_metrics(
            self.per_class_ndcg_scores
        )
        self.f1_per_class, self.f1_aggregated = self._process_per_class_metrics(
            self.per_class_f1_scores
        )
        self.f1_per_class_random, self.f1_aggregated_random = self._process_per_class_metrics(
            self.per_class_f1_scores_random
        )

        # Build and return the result dictionary
        return self._build_result_dict()
    
        

    def name(self):
        return f'Precision@k, Recall@k, F1@k, mAP@k, NDCG@k (k={self.k})'