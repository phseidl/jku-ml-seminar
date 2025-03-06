import torch

class MarginAccuracy:
    def __init__(self, margin_seconds, window_size, sample_rate):
        """
        Initialize the Margin accuracy metric.

        :param margin_seconds: The allowed margin (before/after onset/offset) in seconds.
        :param window_size: The duration of each window in seconds.
        :param sample_rate: The EEG sampling rate (Hz).
        """
        self.margin_windows = margin_seconds // window_size  # Convert seconds to number of windows
        self.total_onsets = 0
        self.correct_onsets = 0
        self.total_offsets = 0
        self.correct_offsets = 0

    def update(self, predictions, labels):
        """
        Update the metric based on batch predictions and labels.

        :param predictions: Model outputs (binary: 1 = seizure detected, 0 = no seizure), shape (batch_size,).
        :param labels: Ground truth labels (binary), shape (batch_size,).
        """
        assert predictions.shape == labels.shape, "Predictions and labels must have the same shape"

        # Find seizure onsets and offsets in window-based labels
        seizure_starts = torch.where((labels[1:] == 1) & (labels[:-1] == 0))[0] + 1
        seizure_ends = torch.where((labels[1:] == 0) & (labels[:-1] == 1))[0] + 1

        # Track total number of onsets and offsets
        self.total_onsets += len(seizure_starts)
        self.total_offsets += len(seizure_ends)

        # Check onset detections
        for start in seizure_starts:
            start_margin = max(0, start - self.margin_windows)
            end_margin = min(len(labels), start + self.margin_windows)

            if torch.any(predictions[start_margin:end_margin] == 1):
                self.correct_onsets += 1

        # Check offset detections
        for end in seizure_ends:
            start_margin = max(0, end - self.margin_windows)
            end_margin = min(len(labels), end + self.margin_windows)

            if torch.any(predictions[start_margin:end_margin] == 1):
                self.correct_offsets += 1

    def compute(self):
        """Compute the final ONSET and OFFSET accuracy."""
        onset_accuracy = self.correct_onsets / self.total_onsets if self.total_onsets > 0 else 0
        offset_accuracy = self.correct_offsets / self.total_offsets if self.total_offsets > 0 else 0
        return onset_accuracy, offset_accuracy

    def reset(self):
        """Reset the metric for a new evaluation run."""
        self.total_onsets = 0
        self.correct_onsets = 0
        self.total_offsets = 0
        self.correct_offsets = 0
