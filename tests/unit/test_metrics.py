import pytest
from text2cypher.finetuning.eval import metrics
import torch

@pytest.fixture
def dummy_data():
    predictions = ["The patient has a mild fever."]
    references = ["The patient has a mild fever."]
    instructions = ["Summarize the conversation about the patient's symptoms."]
    return predictions, references, instructions

@pytest.mark.parametrize("metric_fn", [
    metrics.calculate_rouge,
    metrics.calculate_bleu,
    metrics.calculate_bertscore
])
def test_classical_metrics(metric_fn, dummy_data):
    pred, ref, instr = dummy_data
    score = metric_fn(pred, ref, instr)
    assert isinstance(score, float)
    assert 0 <= score <= 1

@pytest.mark.parametrize("llm_metric_fn", [
    metrics.calculate_relevance,
    metrics.calculate_factual_consistency,
    metrics.calculate_completeness,
    metrics.calculate_clarity,
    metrics.calculate_conciseness
])
def test_llm_metrics_with_mock(mocker, llm_metric_fn, dummy_data):
    mock_eval = mocker.patch(
        "text2cypher.finetuning.eval.metrics.evaluate_with_grader",
        return_value=4.2
    )
    pred, ref, instr = dummy_data
    score = llm_metric_fn(pred, ref, instr)
    assert score == 4.2
    mock_eval.assert_called_once()

def test_calculate_model_size_in_params():
    class Dummy:
        model = type("Model", (), {"parameters": lambda self: [torch.nn.Parameter(torch.randn(10, 10))]})()
    assert isinstance(metrics.calculate_model_size_in_params(Dummy()), int)
