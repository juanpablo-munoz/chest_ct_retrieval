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

    if cfg["metrics"].get("average_nonzero_triplets", True):
        metrics.append(AverageNonzeroTripletsMetric())
    if cfg["metrics"].get("total_nonzero_triplets", True):
        metrics.append(TotalNonzeroTripletsMetric())
    if cfg["metrics"].get("loss", True):
        metrics.append(Loss())
    if cfg["metrics"].get("ndcg", False):
        metrics.append(NDCG(PROXIMITY_VECTOR_LABELS_FOR_TRAINING, PROXIMITY_CLASS_NAMES))
    if cfg["metrics"].get("recall", False):
        metrics.append(Recall(PROXIMITY_VECTOR_LABELS_FOR_TRAINING, PROXIMITY_CLASS_NAMES))
    if cfg["metrics"].get("all_metrics", True):
        metrics.append(AllMetrics(PROXIMITY_VECTOR_LABELS_FOR_TRAINING, PROXIMITY_CLASS_NAMES))

    return metrics
