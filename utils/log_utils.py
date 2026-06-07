import time

from typing import Dict, Optional


class AverageMeter:
    """
    A utility class for computing running averages of metrics and losses.

    This class maintains running totals and counts for a dictionary of metrics,
    allowing efficient calculation of averages as data streams in.

    Attributes:
        totals (Dict[str, float]): Running totals for each metric
        counts (Dict[str, int]): Number of updates for each metric
        averages (Dict[str, float]): Current running averages for each metric
    """

    def __init__(self, initial_metrics: Dict[str, float] | None = None) -> None:
        """
        Initialize the AverageMeter with optional initial metrics.

        Args:
            initial_metrics (Dict[str, float], optional): Initial metric values.
                If None, starts with zeros. Defaults to None.
        """
        self._totals: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._averages: Dict[str, float] = {}

        if initial_metrics:
            for metric_name, value in initial_metrics.items():
                self._totals[metric_name] = value
                self._counts[metric_name] = 1

    def update(self, metrics_dict: Dict[str, float]) -> None:
        """
        Update running totals and counts with new metric values.

        This method calculates the difference between current and previous
        values to maintain accurate running averages.

        Args:
            metrics_dict (Dict[str, float]): Dictionary containing metric names
                as keys and their current values as values.
        """
        for metric_name, current_value in metrics_dict.items():
            if metric_name not in self._totals:
                # New metric encountered
                self._totals[metric_name] = 0.0
                self._counts[metric_name] = 0
                self._averages[metric_name] = 0.0

            # Calculate the difference and update running total
            self._totals[metric_name] += current_value
            self._counts[metric_name] += 1
            # Calculate average on-demand or cache it
            self._averages[metric_name] = (
                self._totals[metric_name] / self._counts[metric_name]
            )

    def get_average(self) -> Dict[str, float]:
        """
        Get the current running averages for all metrics.

        Returns:
            Dict[str, float]: Dictionary mapping metric names to their running averages.
        """
        return self._averages.copy()

    def reset(self) -> None:
        """
        Reset all running totals, counts, and averages to initial state.

        This clears all accumulated data, allowing the AverageMeter to
        track new metrics from scratch.
        """
        self._totals.clear()
        self._counts.clear()
        self._averages.clear()

    def get_total(self) -> Dict[str, float]:
        """
        Get the current running totals for all metrics.

        Returns:
            Dict[str, float]: Dictionary mapping metric names to their running totals.
        """
        return self._totals.copy()

    def get_count(self) -> Dict[str, int]:
        """
        Get the number of updates for each metric.

        Returns:
            Dict[str, int]: Dictionary mapping metric names to their counts.
        """
        return self._counts.copy()


class Timer:
    """
    A utility class for profiling time between operations.

    This class measures elapsed time between calls to record(),
    allowing for accurate timing without cluttering the main logic.

    Attributes:
        last_time (float): Timestamp of the last record() call
        total_time (float): Cumulative time since last reset
        call_count (int): Number of record() calls after reset
    """

    def __init__(self) -> None:
        """
        Initialize the Timer with a unique time key identifier.

        Args:
            time_key (str): Unique identifier for this timer instance
        """
        self._last_time: float = 0.0
        self._total_time: float = 0.0
        self._call_count: int = 0

    def reset(self) -> None:
        """
        Reset the timer to initial state.

        Clears all accumulated timing data, allowing fresh measurements.
        """
        self._last_time = 0.0
        self._total_time = 0.0
        self._call_count = 0

    def record(self) -> None:
        """
        Record the current time and calculate elapsed time since last record.

        Updates total_time with the elapsed time since the previous call,
        and increments the call counter.
        """

        current_time = time.time()
        if self._call_count > 0:
            elapsed = current_time - self._last_time
            self._total_time += elapsed
        self._last_time = current_time
        self._call_count += 1

    def total_time(self) -> float:
        """
        Get the total accumulated time since reset.

        Returns:
            float: Total time in seconds
        """
        return self._total_time

    def avg_time(self) -> Optional[float]:
        """
        Get the average time per call since reset.

        Returns:
            float: Average time per call in seconds, or None if no calls recorded
        """
        if self._call_count == 0:
            return None
        return self._total_time / self._call_count

    def avg_rate(self) -> Optional[float]:
        """
        Get the average calls recorded per second since reset.

        Returns:
            float: Average calls recorded per second, or None if no calls recorded
        """
        if self._call_count == 0:
            return None
        return self._call_count / self._total_time
