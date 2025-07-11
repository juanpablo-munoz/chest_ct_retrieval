from eval.metrics import (
    Loss,
    AverageNonzeroTripletsMetric,
    TotalNonzeroTripletsMetric,
    NDCG,
    Recall,
    AllMetrics,
)
from datasets.constants import (
    PROXIMITY_VECTOR_LABELS_FOR_TRAINING,
    PROXIMITY_CLASS_NAMES
)

_METRIC_REGISTRY = {
    "Loss": Loss,
    "AverageNonzeroTripletsMetric": AverageNonzeroTripletsMetric,
    "TotalNonzeroTripletsMetric": TotalNonzeroTripletsMetric,
    "NDCG": NDCG,
    "Recall": Recall,
    "AllMetrics": AllMetrics,
}

def load_metrics(cfg):
    metrics = []

    if "average_nonzero_triplets" in cfg["training"]["metrics"]:
        metrics.append(AverageNonzeroTripletsMetric())
    if "total_nonzero_triplets" in cfg["training"]["metrics"]:
        metrics.append(TotalNonzeroTripletsMetric())
    if "loss" in cfg["training"]["metrics"]:
        metrics.append(Loss())
    if "ndcg" in cfg["training"]["metrics"]:
        metrics.append(NDCG(PROXIMITY_VECTOR_LABELS_FOR_TRAINING, PROXIMITY_CLASS_NAMES))
    if "recall" in cfg["training"]["metrics"]:
        metrics.append(Recall(PROXIMITY_VECTOR_LABELS_FOR_TRAINING, PROXIMITY_CLASS_NAMES))
    if "all_metrics" in cfg["training"]["metrics"]:
        metrics.append(AllMetrics(PROXIMITY_VECTOR_LABELS_FOR_TRAINING, PROXIMITY_CLASS_NAMES))

    return metrics
