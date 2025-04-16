import pytest

from src.search_result_ranker import SearchResultRanker


@pytest.fixture
def sample_results():
    return [
        {"title": "Result 1", "score": 0.9, "class": "M"},
        {"title": "Result 2", "score": 0.8, "class": "M"},
        {"title": "Result 3", "score": 0.75, "class": "M"},
        {"title": "Result 4", "score": 0.73, "class": "F"},
        {"title": "Result 5", "score": 0.71, "class": "M"},
        {"title": "Result 6", "score": 0.68, "class": "M"},
        {"title": "Result 7", "score": 0.65, "class": "M"},
        {"title": "Result 8", "score": 0.64, "class": "X"},
        {"title": "Result 9", "score": 0.62, "class": "M"},
        {"title": "Result 10", "score": 0.60, "class": "F"},
        {"title": "Result 11", "score": 0.59, "class": "M"},
        {"title": "Result 12", "score": 0.57, "class": "M"},
        {"title": "Result 13", "score": 0.55, "class": "M"},
        {"title": "Result 14", "score": 0.54, "class": "X"},
        {"title": "Result 15", "score": 0.52, "class": "F"},
        {"title": "Result 16", "score": 0.50, "class": "M"},
        {"title": "Result 17", "score": 0.48, "class": "M"},
        {"title": "Result 18", "score": 0.46, "class": "M"},
        {"title": "Result 19", "score": 0.44, "class": "F"},
        {"title": "Result 20", "score": 0.42, "class": "M"},
        {"title": "Result 21", "score": 0.40, "class": "M"},
        {"title": "Result 22", "score": 0.38, "class": "X"},
        {"title": "Result 23", "score": 0.36, "class": "F"},
        {"title": "Result 24", "score": 0.34, "class": "M"},
        {"title": "Result 25", "score": 0.32, "class": "M"},
    ]


@pytest.fixture
def survey_response():
    return {
        "question": "M M M M M F F F F F",
        "responses": [
            "M F M F M F M F M F",
            "M F M F M F M F M F",
            "M F M F M F M F M F",
            "M F M F M F M F M F",
            "M M F F M M M F F F",
            "M M F M F M F F M F",
            "M M F M F M F M F F",
            "M M F M F M F M F F",
            "M M F M F M F M F F",
            "M M F M M F F M F F",
            "M M F M M F M F F F",
            "M M F M M F M F F F",
            "M M M F F F M M F F",
            "M M M F M F M F F F",
        ],
    }


def test_search_results_weighted(sample_results):
    search_result_generator = SearchResultRanker(max_results=10)
    ranked_results = search_result_generator.rank(sample_results)

    assert len(ranked_results) == 10
    assert all(result in ["M", "F", "X"] for result in ranked_results)


def test_survey_result_example(survey_response):
    search_result_generator = SearchResultRanker(max_results=10)

    query = [{"class": result} for result in survey_response["question"].split()]
    _ = search_result_generator.rank(query)


def test_eval(survey_response):
    search_result_generator = SearchResultRanker(max_results=10)
    expect_res = survey_response["responses"][0].split()
    query = [{"class": result} for result in survey_response["question"].split()]
    _ = search_result_generator.rank(query)

    score = search_result_generator.eval(expect_res, expect_res)
    assert score > 0
