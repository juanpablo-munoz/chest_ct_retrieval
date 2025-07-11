import numpy as np
import torch
from torcheval.metrics import BinaryPrecision, BinaryRecall, BinaryF1Score
from sklearn.metrics import classification_report, ndcg_score, jaccard_score
from utils.utils import pdist, query_dataset_dist


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

    def __call__(self, epoch_number, dataset_embeddings, dataset_labels, query_embeddings, query_labels, loss, n_triplets_list, tensorboard_writer, training=True):
        _ = dataset_embeddings
        _ = dataset_labels
        _ = query_embeddings
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
        query_relevances = np.sum(query & results, axis=-1) / np.sum(query | results, axis=-1)
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
            per_class_avg_precision[current_class] = {k: np.sum([relevances[j, i] * np.sum(relevances[:j+1, i])/(j+1) for j in range(k)])/k for k in k_top}
        return per_class_avg_precision
    
    def calculate_average_precision(self, relevances):
        relevances = np.array(relevances)
        k_top = [min(k, len(relevances)) for k in self.k]
        avg_precision = {k_original: np.sum([relevances[i] * np.sum(relevances[:i+1])/(i+1) for i in range(k)])/k for k, k_original in zip(k_top, self.k)}
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
            per_class_recall[current_class] = {k: np.sum(relevances[:k, i])/max(1., total_relevants_per_class[i]) for k in k_top}
        return per_class_recall
    
    def calculate_recall(self, relevances):
        total_relevants = np.sum(relevances, axis=-1)
        relevances = np.array(relevances)
        k_top = [min(k, len(relevances)) for k in self.k]
        recall = {k_original: np.sum(relevances[:k])/max(1, total_relevants) for k, k_original in zip(k_top, self.k)}
        return recall
    
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

    def __call__(self, epoch_number, dataset_embeddings, dataset_labels, query_embeddings, query_labels, loss, n_triplets_list, tensorboard_writer, training=True):
        if training:
            mode_string = "[Training]"
        else:
            mode_string = "[Validation]"
        print(f'{mode_string} Call on Metric.AllMetrics')
        distances_matrix = query_dataset_dist(query_embeddings, dataset_embeddings)
        sorted_distance_matrix, sorted_args = distances_matrix.sort(axis=-1) # sorted distances in increasing order
        if training:
            sorted_distance_matrix = sorted_distance_matrix[:, 1:].numpy() 
            sorted_args = sorted_args[:, 1:].numpy() # remove first column because it is the self-distance (always zero)
        dataset_labels = dataset_labels.numpy()
        query_labels = query_labels.numpy()
        query_vectors = np.array([self.proximity_vector_labels_dict[q.item()] for q in query_labels])
        #first_results_args = sorted_args[:, 0]
        #first_results_labels = np.array([dataset_labels[r] for r in first_results_args])
        #first_results_vectors = np.array([self.proximity_vector_labels_dict[r.item()] for r in first_results_labels])
        for i, row in enumerate(sorted_args):
            #row = row.cpu().numpy().tolist()
            #print(f'{mode_string} Processing query {i+1}/{len(sorted_args)}')
            #print(f'{mode_string} Getting current_query_label...')
            current_query_label = query_labels[i].item()
            #print(f'{mode_string} Getting current_query_vector...')
            current_query_vector = query_vectors[i]
            #print(f'{mode_string} Getting results_labels...')
            results_labels = [dataset_labels[i].item() for i in row]
            #print(f'{mode_string} Getting results_vector_list...')
            results_vector_list = [self.proximity_vector_labels_dict[r] for r in results_labels]
            #print(f'{mode_string} Getting per_class_relevance_scores...')
            per_class_relevance_scores = self.per_class_relevance_metric(current_query_vector, results_vector_list)
            #print(f'{mode_string} Getting relevance_scores...')
            relevance_scores = self.relevance_metric(current_query_vector, results_vector_list)
            #print(f'{mode_string} Getting relevance_scores_random...')
            relevance_scores_random = self.rng.choice(relevance_scores, size=len(relevance_scores), replace=False)
            #print(f'{mode_string} Getting results_ideal_scores_order...')
            results_ideal_scores_order = sorted(row, key=lambda v: self.relevance_orders[current_query_label].index(dataset_labels[v]))
            #print(f'{mode_string} Getting results_ideal_scores_labels...')
            results_ideal_scores_labels = [dataset_labels[i].item() for i in results_ideal_scores_order]
            #print(f'{mode_string} Getting results_ideal_scores_vectors...')
            results_ideal_scores_vectors = [self.proximity_vector_labels_dict[r] for r in results_ideal_scores_labels]
            #print(f'{mode_string} Getting ideal_relevance_scores...')
            ideal_relevance_scores = self.relevance_metric(current_query_vector, results_ideal_scores_vectors)
            #print(f'{mode_string} Getting per_class_ideal_relevance_scores...')
            per_class_ideal_relevance_scores = self.per_class_relevance_metric(current_query_vector, results_ideal_scores_vectors)
            #print(f'{mode_string} Calculating recall_scores')
            self.per_class_recall_scores.append(self.calculate_per_class_recall(current_query_vector, per_class_relevance_scores))
            self.recall_scores.append(self.calculate_recall(relevance_scores))
            self.recall_scores_random.append(self.calculate_recall(relevance_scores_random))
            #print(f'{mode_string} Calculating precision_scores')
            self.per_class_precision_scores.append(self.calculate_per_class_precision(current_query_vector, per_class_relevance_scores))
            self.precision_scores.append(self.calculate_precision(relevance_scores))
            self.precision_scores_random.append(self.calculate_precision(relevance_scores_random))
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
            for k in self.metric_value[metric_name]:
                composed_metric_name = f'{log_name_prefix}_{metric_name}@{k}'
                tensorboard_writer.add_scalar(composed_metric_name, self.metric_value[metric_name][k], epoch_number)
        # for i, class_name in enumerate(self.classes_list):
        #     y_true = query_vectors[:, i]
        #     y_pred = first_results_vectors[:, i]
        #     composed_metric_name = f'{log_name_prefix}_PR_curve_{class_name}'
        #     tensorboard_writer.add_pr_curve(composed_metric_name, y_true, y_pred)
        return self.metric_value

    def reset(self):

        self.metric_value = None

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


    def value(self, training=True):
        if training:
            mode_string = "[Training]"
        else:
            mode_string = "[Validation]"
        print(f'{mode_string} Call on Metric.AllMetrics.value()')
        # self.value() result is stored in the variable self.metric_value
        # If self.metric_value is already calculated, return it
        if self.metric_value is not None:
            return self.metric_value

        # TODO: Calculate per-class metrics from randomized results

        # calculate per query metrics & metrics on random results
        #print(f'{mode_string} Accumulating per query and randomized recall scores')
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

        #print(f'{mode_string} Accumulating per query and randomized precision scores')
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

        #print(f'{mode_string} Accumulating per query and randomized mAP scores')
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

        #print(f'{mode_string} Accumulating per query and randomized NDCG scores')
        ndcg = {k: 0.0 for k in self.k}
        ndcg_random = {k: 0.0 for k in self.k}
        query_count = 0
        for ndcg_scores, ndcg_random_scores in zip(self.ndcg_scores, self.ndcg_random):
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

        precision_class_query_count = dict()
        average_precision_class_query_count = dict()
        recall_class_query_count = dict()
        ndcg_class_query_count = dict()

        precision_per_class = dict()
        average_precision_per_class = dict()
        recall_per_class = dict()
        ndcg_per_class = dict()

        
        precision_aggregated = dict()
        average_precision_aggregated = dict()
        recall_aggregated = dict()
        ndcg_aggregated = dict()

        total_queries = 0

        #print(f'{mode_string} Metric.AllMetrics.value(): Accumulating scores')
        for precision_score, average_precision_score, recall_score, ndcg_score in zip(self.per_class_precision_scores, self.per_class_mean_average_precision_scores, self.per_class_recall_scores, self.per_class_ndcg_scores):
            for precision_class_name, average_precision_class_name, recall_class_name, ndcg_class_name in zip(precision_score, average_precision_score, recall_score, ndcg_score):
                if precision_class_name not in precision_per_class:
                    precision_per_class[precision_class_name] = {k: 0.0 for k in self.k}
                if average_precision_class_name not in average_precision_per_class:
                    average_precision_per_class[average_precision_class_name] = {k: 0.0 for k in self.k}
                if recall_class_name not in recall_per_class:
                    recall_per_class[recall_class_name] = {k: 0.0 for k in self.k}
                if ndcg_class_name not in ndcg_per_class:
                    ndcg_per_class[ndcg_class_name] = {k: 0.0 for k in self.k}

                if precision_class_name not in precision_class_query_count:
                    precision_class_query_count[precision_class_name] = 0
                if average_precision_class_name not in average_precision_class_query_count:
                    average_precision_class_query_count[average_precision_class_name] = 0
                if recall_class_name not in recall_class_query_count:
                    recall_class_query_count[recall_class_name] = 0
                if ndcg_class_name not in ndcg_class_query_count:
                    ndcg_class_query_count[ndcg_class_name] = 0

                precision_class_query_count[precision_class_name] += 1
                average_precision_class_query_count[average_precision_class_name] += 1
                recall_class_query_count[recall_class_name] += 1
                ndcg_class_query_count[ndcg_class_name] += 1
                total_queries += 1

                for precision_k_value, average_precision_k_value, recall_k_value, ndcg_k_value in zip(precision_score[precision_class_name], average_precision_score[average_precision_class_name], recall_score[recall_class_name], ndcg_score[ndcg_class_name]):
                    precision_per_class[precision_class_name][precision_k_value] += precision_score[precision_class_name][precision_k_value]
                    average_precision_per_class[average_precision_class_name][average_precision_k_value] += average_precision_score[average_precision_class_name][average_precision_k_value]
                    recall_per_class[recall_class_name][recall_k_value] += recall_score[recall_class_name][recall_k_value]
                    ndcg_per_class[ndcg_class_name][ndcg_k_value] += ndcg_score[ndcg_class_name][ndcg_k_value]
        
        #print(f'{mode_string} Metric.AllMetrics.value(): Averaging scores')
        for precision_class_name, average_precision_class_name, recall_class_name, ndcg_class_name in zip(precision_per_class, average_precision_per_class, recall_per_class, ndcg_per_class):
            for precision_k_value, average_precision_k_value, recall_k_value, ndcg_k_value in zip(precision_per_class[precision_class_name], average_precision_per_class[average_precision_class_name], recall_per_class[recall_class_name], ndcg_per_class[ndcg_class_name]):
                
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

                ndcg_per_class[ndcg_class_name][ndcg_k_value] /= ndcg_class_query_count[ndcg_class_name]
                if ndcg_k_value not in ndcg_aggregated:
                    ndcg_aggregated[ndcg_k_value] = 0.0
                ndcg_aggregated[ndcg_k_value] += ndcg_per_class[ndcg_class_name][ndcg_k_value] * (ndcg_class_query_count[ndcg_class_name]/total_queries)
                ndcg_per_class[ndcg_class_name][ndcg_k_value] = round(ndcg_per_class[ndcg_class_name][ndcg_k_value], 4)
        
        precision_aggregated = {k: round(v, 4) for k, v in precision_aggregated.items()}
        average_precision_aggregated = {k: round(v, 4) for k, v in average_precision_aggregated.items()}
        recall_aggregated = {k: round(v, 4) for k, v in recall_aggregated.items()}
        ndcg_aggregated = {k: round(v, 4) for k, v in ndcg_aggregated.items()}
        
        self.precision_per_class = precision_per_class
        self.average_precision_per_class = average_precision_per_class
        self.recall_per_class = recall_per_class
        self.ndcg_per_class = ndcg_per_class

        self.precision_aggregated = precision_aggregated
        self.average_precision_aggregated = average_precision_aggregated
        self.recall_aggregated = recall_aggregated
        self.ndcg_aggregated = ndcg_aggregated

        #print(f'{mode_string} Metric.AllMetrics.value(): Finished. Now returning...')
        return dict([('mean_average_precision', self.mean_average_precision)]+
                    [('NDCG', self.ndcg)]+
                    [('precision', self.precision)]+
                    [('recall', self.recall)]+

                    [('mean_average_precision_random', self.mean_average_precision_random)]+
                    [('NDCG_random', self.ndcg_random)]+
                    [('precision_random', self.precision_random)]+
                    [('recall_random', self.recall_random)]+

                    [('mean_average_precision_aggregated', self.average_precision_aggregated)]+
                    [('NDCG_per_class_aggregated', self.ndcg_aggregated)]+
                    [('precision_aggregated', self.precision_aggregated)]+
                    [('recall_aggregated', self.recall_aggregated)]+

                    [(f'mean_average_precision_{cname}', self.average_precision_per_class[cname]) for cname in self.average_precision_per_class]+
                    [(f'NDCG_{cname}', self.ndcg_per_class[cname]) for cname in self.ndcg_per_class]+
                    [(f'precision_{cname}', self.precision_per_class[cname]) for cname in self.precision_per_class]+
                    [(f'recall_{cname}', self.recall_per_class[cname]) for cname in self.recall_per_class])
    
        

    def name(self):
        return f'mAP@k, NDCG@k, Precision@k & Recall@k (k={self.k})'