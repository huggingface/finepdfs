import logging
import random
import statistics
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, model_validator


class DatasetStatistics(BaseModel):
    total: int

    mean: float
    median: float
    std: float

    bins: List[float]
    hist: List[float]

    @model_validator(mode="after")
    def check_bins_and_hist_lengths(cls, values):
        if len(values.bins) != len(values.hist) + 1:
            raise ValueError("`bins` must have exactly one more element than `hist`.")
        return values

    def to_table(self, metric_name: str) -> Tuple[List[List[str]], List[str]]:
        headers = [
            f"{metric_name}",
            "prob [%]",
            "acc [%]",
            "1-acc [%]",
            "total",
        ]
        cumsum: float = 0.0

        table = []
        if self.total > 0:
            for i in range(len(self.bins) - 1):
                table.append(
                    [
                        f"({self.bins[i + 0]:.3f}, {self.bins[i + 1]:.3f}]",
                        f"{100.0 * float(self.hist[i]) / float(self.total):.2f}",
                        f"{100.0 * cumsum:.2f}",
                        f"{100.0 * (1.0-cumsum):.2f}",
                        f"{self.hist[i]}",
                    ]
                )
                cumsum += float(self.hist[i]) / float(self.total)
        return table, headers

    def save_histogram(self, figname: Path, name: str = ""):
        # Calculate bin widths
        bin_widths = [
            self.bins[i + 1] - self.bins[i] for i in range(len(self.bins) - 1)
        ]
        bin_middle = [
            (self.bins[i + 1] + self.bins[i]) / 2.0 for i in range(len(self.bins) - 1)
        ]

        # Plot histogram
        fignum = int(1000 * random.random())
        plt.figure(fignum)
        plt.bar(bin_middle, self.hist, width=bin_widths, edgecolor="black")

        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.title(
            f"{name} (mean: {self.mean:.2f}, median: {self.median:.2f}, std: {self.std:.2f}, total: {self.total})"
        )

        logging.info(f"Saving figure to {figname}")
        plt.savefig(figname)


def compute_stats(values: List[float]) -> DatasetStatistics:
    total: int = len(values)

    mean: float = statistics.mean(values) if len(values) > 0 else -1
    median: float = statistics.median(values) if len(values) > 0 else -1
    std: float = statistics.stdev(values) if len(values) > 0 else -1
    logging.info(f"total: {total}, mean: {mean}, median: {median}, std: {std}")

    # Compute the histogram with 20 bins between 0 and 1
    hist, bins = np.histogram(values, bins=20, range=(0, 1))
    logging.info(f"#-hist: {len(hist)}, #-bins: {len(bins)}")

    return DatasetStatistics(
        total=total, mean=mean, median=median, std=std, hist=hist, bins=bins
    )
