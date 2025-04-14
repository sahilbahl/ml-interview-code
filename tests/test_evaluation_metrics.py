import numpy as np
import pytest

from src.metrics import accuracy_score
from src.metrics import confusion_matrix
from src.metrics import f1_score
from src.metrics import precision_score
from src.metrics import recall_score


class TestBinaryClassificationMetrics:
    @pytest.fixture
    def binary_data(self):
        """Simple binary classification dataset with clear predictions"""
        # Perfect predictions
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        return y_true, y_pred

    @pytest.fixture
    def binary_data_with_errors(self):
        """Binary classification with some errors"""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 0])
        return y_true, y_pred

    def test_perfect_accuracy(self, binary_data):
        """Test accuracy with perfect predictions"""
        y_true, y_pred = binary_data
        assert accuracy_score(y_true, y_pred) == 1.0

    def test_imperfect_accuracy(self, binary_data_with_errors):
        """Test accuracy with imperfect predictions"""
        y_true, y_pred = binary_data_with_errors
        assert accuracy_score(y_true, y_pred) == 0.625  # 5/8 correct

    def test_binary_precision(self, binary_data_with_errors):
        """Test precision for binary classification"""
        y_true, y_pred = binary_data_with_errors
        # Class 0: 3 correctly predicted out of 5 predictions = 0.6
        # Class 1: 2 correctly predicted out of 3 predictions ≈ 0.667
        assert precision_score(y_true, y_pred, average=None) == pytest.approx(
            [0.6, 0.667], abs=0.001
        )
        assert precision_score(y_true, y_pred, average="macro") == pytest.approx(
            0.633, abs=0.001
        )
        assert precision_score(y_true, y_pred, average="weighted") == pytest.approx(
            0.633, abs=0.001
        )

    def test_binary_recall(self, binary_data_with_errors):
        """Test recall for binary classification"""
        y_true, y_pred = binary_data_with_errors
        # Class 0: 3 correctly predicted out of 4 actual
        # Class 1: 2 correctly predicted out of 4 actual
        assert recall_score(y_true, y_pred, average=None) == pytest.approx([0.75, 0.5])
        assert recall_score(y_true, y_pred, average="macro") == pytest.approx(0.625)
        assert recall_score(y_true, y_pred, average="weighted") == pytest.approx(0.625)

    def test_binary_f1(self, binary_data_with_errors):
        """Test F1 score for binary classification"""
        y_true, y_pred = binary_data_with_errors
        assert f1_score(y_true, y_pred, average=None) == pytest.approx(
            [0.667, 0.571], abs=0.001
        )
        assert f1_score(y_true, y_pred, average="macro") == pytest.approx(
            0.619, abs=0.001
        )
        assert f1_score(y_true, y_pred, average="weighted") == pytest.approx(
            0.619, abs=0.001
        )

    def test_binary_confusion_matrix(self, binary_data_with_errors):
        """Test confusion matrix for binary classification"""
        y_true, y_pred = binary_data_with_errors
        expected_cm = np.array(
            [
                [3, 1],  # 3 true negatives, 1 false positive
                [2, 2],  # 2 false negatives, 2 true positives
            ]
        )
        pred_matrix = confusion_matrix(y_true, y_pred)
        assert np.array_equal(pred_matrix, expected_cm)


class TestMultiClassMetrics:
    @pytest.fixture
    def multi_class_data(self):
        """Multi-class classification dataset"""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 2])
        y_pred = np.array([0, 2, 1, 0, 1, 2, 0, 1])
        return y_true, y_pred

    @pytest.fixture
    def multi_class_data_non_numeric(self):
        """Multi-class classification dataset"""
        y_true = np.array(["A", "B", "C", "A", "B", "C", "A", "C"])
        y_pred = np.array(["A", "C", "B", "A", "B", "C", "A", "B"])
        return y_true, y_pred

    def test_multi_class_accuracy(self, multi_class_data):
        """Test accuracy for multi-class classification"""
        y_true, y_pred = multi_class_data
        # 5 correct predictions out of 8
        assert accuracy_score(y_true, y_pred) == 0.625

    def test_multi_class_precision(self, multi_class_data):
        """Test precision for multi-class classification"""
        y_true, y_pred = multi_class_data
        # Class 0: 3 correctly predicted out of 3 predictions = 1.0
        # Class 1: 1 correctly predicted out of 3 predictions = 0.333
        # Class 2: 1 correctly predicted out of 2 predictions = 0.5
        expected_precision = [1.0, 1 / 3, 0.5]
        assert precision_score(y_true, y_pred, average=None) == pytest.approx(
            expected_precision
        )

        # Macro average: (1.0 + 1/3 + 0.5) / 3 = 0.611
        assert precision_score(y_true, y_pred, average="macro") == pytest.approx(
            0.611, abs=0.01
        )

        # Weighted average: (1.0*3 + 1/3*2 + 0.5*3) / 8 = 0.646
        assert precision_score(y_true, y_pred, average="weighted") == pytest.approx(
            0.646, abs=0.01
        )

    def test_multi_class_recall(self, multi_class_data):
        """Test recall for multi-class classification"""
        y_true, y_pred = multi_class_data
        # Class 0: 3 correctly predicted out of 3 actual = 1.0
        # Class 1: 1 correctly predicted out of 2 actual = 0.5
        # Class 2: 1 correctly predicted out of 3 actual = 0.333
        expected_recall = [1.0, 0.5, 1 / 3]
        assert recall_score(y_true, y_pred, average=None) == pytest.approx(
            expected_recall
        )

        # Macro average: (1.0 + 0.5 + 1/3) / 3 = 0.611
        assert recall_score(y_true, y_pred, average="macro") == pytest.approx(
            0.611, abs=0.01
        )

        # Weighted average: (1.0*3 + 0.5*2 + 1/3*3) / 8 = 0.625
        assert recall_score(y_true, y_pred, average="weighted") == pytest.approx(
            0.625, abs=0.01
        )

    def test_multi_class_f1(self, multi_class_data):
        """Test F1 score for multi-class classification"""
        y_true, y_pred = multi_class_data
        # Class 0: 2 * (1.0 * 1.0) / (1.0 + 1.0) = 1.0
        # Class 1: 2 * (1/3 * 0.5) / (1/3 + 0.5) ≈ 0.4
        # Class 2: 2 * (0.5 * 1/3) / (0.5 + 1/3) ≈ 0.4
        expected_f1 = [1.0, 0.4, 0.4]
        assert f1_score(y_true, y_pred, average=None) == pytest.approx(
            expected_f1, abs=0.01
        )

        # Macro average: (1.0 + 0.4 + 0.4) / 3 = 0.6
        assert f1_score(y_true, y_pred, average="macro") == pytest.approx(0.6, abs=0.01)

        # Weighted average: (1.0*3 + 0.4*2 + 0.4*3) / 8 = 0.625
        assert f1_score(y_true, y_pred, average="weighted") == pytest.approx(
            0.625, abs=0.01
        )

    def test_multi_class_confusion_matrix(self, multi_class_data):
        """Test confusion matrix for multi-class classification"""
        y_true, y_pred = multi_class_data
        expected_cm = np.array(
            [
                [3, 0, 0],  # Class 0: 3 true predictions, 0 for class 1, 0 for class 2
                [
                    0,
                    1,
                    1,
                ],  # Class 1: 0 for class 0, 1 true prediction, 1 false for class 2
                [
                    0,
                    2,
                    1,
                ],  # Class 2: 0 for class 0, 2 false for class 1, 1 true prediction
            ]
        )
        assert np.array_equal(confusion_matrix(y_true, y_pred), expected_cm)

    def test_multi_class_confusion_matrix_non_numeric(
        self, multi_class_data_non_numeric
    ):
        """Test confusion matrix for multi-class classification"""
        y_true, y_pred = multi_class_data_non_numeric
        expected_cm = np.array(
            [
                [3, 0, 0],  # Class 0: 3 true predictions, 0 for class 1, 0 for class 2
                [
                    0,
                    1,
                    1,
                ],  # Class 1: 0 for class 0, 1 true prediction, 1 false for class 2
                [
                    0,
                    2,
                    1,
                ],  # Class 2: 0 for class 0, 2 false for class 1, 1 true prediction
            ]
        )
        assert np.array_equal(confusion_matrix(y_true, y_pred), expected_cm)


class TestEdgeCases:
    def test_zero_division_precision(self):
        """Test handling division by zero in precision calculation"""
        # Class 1 has no predictions, which would cause division by zero
        y_true = np.array([0, 0, 0, 0, 1])
        y_pred = np.array([0, 0, 0, 0, 0])

        # Default behavior should be to return 0
        assert precision_score(y_true, y_pred, average=None)[1] == 0.0

        # Test with zero_division=1.0
        assert (
            precision_score(y_true, y_pred, average=None, zero_division=1.0)[1] == 1.0
        )

    def test_zero_division_recall(self):
        """Test handling division by zero in recall calculation"""
        # Class 1 has no actual samples, which would cause division by zero
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 1, 1])

        # Default behavior should be to return 0
        assert recall_score(y_true, y_pred, average=None)[1] == 0.0

        # Test with zero_division=1.0
        assert recall_score(y_true, y_pred, average=None, zero_division=1.0)[1] == 1.0

    def test_empty_arrays(self):
        """Test behavior with empty arrays"""
        y_true = np.array([])
        y_pred = np.array([])

        # Should return NaN for empty arrays
        assert np.isnan(accuracy_score(y_true, y_pred))

        # Confusion matrix should be empty
        assert confusion_matrix(y_true, y_pred).size == 0

    def test_different_length_arrays(self):
        """Test that different length arrays raise ValueError"""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1])

        with pytest.raises(ValueError):
            accuracy_score(y_true, y_pred)
