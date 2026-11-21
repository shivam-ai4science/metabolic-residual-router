import unittest

import numpy as np
from sklearn import metrics

from src.lightning_module import average_precision, pr_auc, roc_auc


class RankMetricTests(unittest.TestCase):
    def test_ties_match_official_sklearn_style_metrics(self):
        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        probabilities = np.array([1.0, 1.0, 0.8, 0.8, 0.5, 0.5, 0.0, 0.0])

        fpr, tpr, _ = metrics.roc_curve(labels, probabilities)
        precision, recall, _ = metrics.precision_recall_curve(labels, probabilities)
        self.assertAlmostEqual(roc_auc(labels, probabilities), metrics.auc(fpr, tpr))
        self.assertAlmostEqual(pr_auc(labels, probabilities), metrics.auc(recall, precision))
        self.assertAlmostEqual(
            average_precision(labels, probabilities),
            metrics.average_precision_score(labels, probabilities),
        )

    def test_metrics_are_invariant_to_order_inside_ties(self):
        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        probabilities = np.array([1.0, 1.0, 0.8, 0.8, 0.5, 0.5, 0.0, 0.0])
        permutation = np.array([1, 0, 3, 2, 5, 4, 7, 6])

        for metric in (roc_auc, pr_auc, average_precision):
            self.assertAlmostEqual(
                metric(labels, probabilities),
                metric(labels[permutation], probabilities[permutation]),
            )


if __name__ == "__main__":
    unittest.main()
