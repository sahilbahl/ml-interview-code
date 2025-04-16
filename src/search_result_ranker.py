from collections import deque
from typing import Dict
from typing import List

import numpy as np


class SearchResultRanker:
    def __init__(self, max_results: int = 10):
        self.max_results = max_results

    def eval(self, expect_res: List[str], actual_res: List[str]) -> float:
        """
        Evaluate the ranker by comparing the expected and actual results.
        """
        if len(expect_res) != len(actual_res):
            raise ValueError("Expected and actual results must be of the same length.")

        # Count the number of matches
        exact_order_match = np.sum(
            np.where(np.array(expect_res) == np.array(actual_res), 1, 0)
        )

        exact_order_match_score = exact_order_match / len(expect_res)
        return exact_order_match_score

    def rank(self, search_results: list[dict]) -> list[dict]:
        """
        Based on the class, return an equal num of resutls for each class based on the max_results

        Args:
            search_results (_type_): ranked search results
        """
        # generate a mapping of class to results
        result_by_class: Dict[str, deque] = {}
        for result in search_results:
            result_class = result["class"]
            if result_class not in result_by_class:
                result_by_class[result_class] = deque()
            result_by_class[result_class].append(result)

        final_results: List = []
        while len(final_results) < self.max_results:
            for cl in sorted(result_by_class.keys()):

                if len(result_by_class[cl]) == 0:
                    continue

                if len(final_results) >= self.max_results:
                    break

                result = result_by_class[cl].popleft()
                final_results.append(result["class"])

        return final_results
