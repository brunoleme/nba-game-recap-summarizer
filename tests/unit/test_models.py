import pytest
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from nba_game_recap_summarizer.finetuning.models.llama_model import LlamaRecapSummarizationModel


@pytest.fixture
def default_llama_model() -> LlamaRecapSummarizationModel:
    return LlamaRecapSummarizationModel(
        model_name="hf-internal-testing/tiny-random-LlamaForCausalLM",
        model_type="llama",
        use_quantization=False,
    )

@pytest.mark.parametrize(
    "model_name, model_type, use_quantization",
    [
        ("hf-internal-testing/tiny-random-LlamaForCausalLM", "llama", False),
        ("hf-internal-testing/tiny-random-LlamaForCausalLM", "llama", True),
        ("llama-nonexistent", "llama", False),
    ]
)
def test_llama_model_initialization(model_name, model_type, use_quantization):
    if model_name == "llama-nonexistent":
        with pytest.raises(Exception):
            LlamaRecapSummarizationModel(
                model_name=model_name,
                model_type=model_type,
                use_quantization=use_quantization
            )
    else:
        model = LlamaRecapSummarizationModel(
            model_name=model_name,
            model_type=model_type,
            use_quantization=use_quantization
        )
        assert model.model_name == model_name
        assert model.model_type == model_type
        assert hasattr(model, "model")
        assert hasattr(model, "tokenizer")



def test_model_summarize_recap(default_llama_model) -> None:
    game_recap = "Lakers beats Suns in the overtime with a Bryant game winner."
    game_recap_summary = default_llama_model.summarize_recap(game_recap, max_length=50)
    assert isinstance(game_recap_summary, str)
    assert len(game_recap_summary) > 0

class DummyDataset(Dataset):
    def __init__(self, tokenizer):
        self.enc = tokenizer("summarize: Tonight", return_tensors="pt", padding=True)
    def __len__(self):
        return 2
    def __getitem__(self, idx):
        return {
            "input_ids": self.enc["input_ids"].squeeze(0),
            "attention_mask": self.enc["attention_mask"].squeeze(0),
        }

def test_model_summarize_recaps(default_llama_model) -> None:
    dataset = DummyDataset(default_llama_model.tokenizer)
    dataloader = DataLoader(dataset, batch_size=1)

    game_recap_summaries = default_llama_model.summarize_recaps(dataloader, max_length=20)
    assert isinstance(game_recap_summaries, list)
    assert all(isinstance(n, str) for n in game_recap_summaries)

def test_summarize_recap_with_empty_input(default_llama_model):
    game_recap_summary = default_llama_model.summarize_recap("", max_length=50)
    assert isinstance(game_recap_summary, str)

@pytest.mark.parametrize("peft_method", ["lora", "prompt_tuning"])
def test_llama_model_with_peft(peft_method):
    model = LlamaRecapSummarizationModel(
        model_name="hf-internal-testing/tiny-random-LlamaForCausalLM",
        model_type="llama",
        peft_method=peft_method,
        use_quantization=False
    )

    assert model.peft_method == peft_method
    assert model.peft_config is not None
    assert hasattr(model.model, "base_model")

    game_recap = "Lakers beats Suns in the overtime with a Bryant game winner."
    game_recap_summary = model.summarize_recap(game_recap, max_length=50)
    assert isinstance(game_recap_summary, str)
    assert len(game_recap_summary) > 0
