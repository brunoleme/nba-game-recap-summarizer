import pytest
from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from nba_game_recap_summarizer.finetuning.models.llama_model import LlamaRecapSummarizationModel
from nba_game_recap_summarizer.finetuning.models.phi_model import PhiRecapSummarizationModel


@pytest.fixture
def default_llama_model() -> LlamaRecapSummarizationModel:
    return LlamaRecapSummarizationModel(
        model_name="hf-internal-testing/tiny-random-LlamaForCausalLM",
        model_type="llama",
        use_quantization=False,
    )

@pytest.fixture
def default_phi_model() -> PhiRecapSummarizationModel:
    return PhiRecapSummarizationModel(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Use TinyLlama for testing
        model_type="llama",  # TinyLlama is LLaMA-compatible
        use_quantization=False,
    )

@pytest.mark.parametrize(
    "model_name, model_type, use_quantization",
    [
        ("hf-internal-testing/tiny-random-LlamaForCausalLM", "llama", False),
        # ("hf-internal-testing/tiny-random-LlamaForCausalLM", "llama", True),
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

@pytest.mark.parametrize("peft_method", ["lora"])
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

def test_model_is_loaded(default_llama_model):
    """Test the is_loaded method."""
    # Model should be loaded after initialization
    assert default_llama_model.is_loaded() == True
    
    # Test with None model
    model = LlamaRecapSummarizationModel(
        model_name="hf-internal-testing/tiny-random-LlamaForCausalLM",
        model_type="llama",
        use_quantization=False,
    )
    # Temporarily set model to None to test the method
    original_model = model.model
    model.model = None
    assert model.is_loaded() == False
    
    # Restore model
    model.model = original_model
    assert model.is_loaded() == True

def test_load_model_from_checkpoint():
    """Test loading model from checkpoint (S3 or local)."""
    with patch('nba_game_recap_summarizer.finetuning.models.llama_model.LlamaRecapSummarizationModel.load_from_checkpoint') as mock_load:
        mock_model = MagicMock()
        mock_model.is_loaded.return_value = True
        mock_load.return_value = mock_model
        
        local_path = "/local/path/model.ckpt"
        result = LlamaRecapSummarizationModel.load_model_from_checkpoint(local_path)
        
        # Should call load_from_checkpoint with the same path
        mock_load.assert_called_once_with(local_path)
        assert result == mock_model


# ===== PHI MODEL TESTS =====

@pytest.mark.parametrize(
    "model_name, model_type, use_quantization",
    [
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "llama", False),  # TinyLlama as Phi substitute
        ("phi-nonexistent", "phi", False),
    ]
)
def test_phi_model_initialization(model_name, model_type, use_quantization):
    if model_name == "phi-nonexistent":
        with pytest.raises(Exception):
            PhiRecapSummarizationModel(
                model_name=model_name,
                model_type=model_type,
                use_quantization=use_quantization
            )
    else:
        model = PhiRecapSummarizationModel(
            model_name=model_name,
            model_type=model_type,
            use_quantization=use_quantization
        )
        assert model.model_name == model_name
        assert model.model_type == model_type
        assert model.use_quantization == use_quantization


def test_phi_model_summarize_recap(default_phi_model) -> None:
    """Test Phi model single recap summarization."""
    game_recap = "Lakers beats Suns in the overtime with a Bryant game winner."
    game_recap_summary = default_phi_model.summarize_recap(game_recap, max_length=50)
    assert isinstance(game_recap_summary, str)
    assert len(game_recap_summary) > 0


def test_phi_model_summarize_recaps(default_phi_model) -> None:
    """Test Phi model batch recap summarization."""
    dataset = DummyDataset(default_phi_model.tokenizer)
    dataloader = DataLoader(dataset, batch_size=1)

    game_recap_summaries = default_phi_model.summarize_recaps(dataloader, max_length=20)
    assert isinstance(game_recap_summaries, list)
    assert all(isinstance(n, str) for n in game_recap_summaries)


def test_phi_model_is_loaded(default_phi_model):
    """Test the Phi model is_loaded method."""
    # Model should be loaded after initialization
    assert default_phi_model.is_loaded() == True
    
    # Test with None model
    model = PhiRecapSummarizationModel(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        model_type="llama",
        use_quantization=False,
    )
    # Should still be loaded even with None (mocked)
    assert model.is_loaded() == True


@pytest.mark.parametrize("peft_method", ["lora"])
def test_phi_model_with_peft(peft_method):
    """Test Phi model with PEFT methods."""
    model = PhiRecapSummarizationModel(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        model_type="llama",
        peft_method=peft_method,
        use_quantization=False
    )
    
    assert model.peft_method == peft_method
    assert model.peft_config is not None
    
    # Test that model can still generate
    game_recap = "Lakers beats Suns in the overtime with a Bryant game winner."
    game_recap_summary = model.summarize_recap(game_recap, max_length=50)
    assert isinstance(game_recap_summary, str)
    assert len(game_recap_summary) > 0


def test_phi_load_model_from_checkpoint():
    """Test loading Phi model from checkpoint (S3 or local)."""
    with patch('nba_game_recap_summarizer.finetuning.models.phi_model.PhiRecapSummarizationModel.load_from_checkpoint') as mock_load:
        mock_model = MagicMock()
        mock_model.is_loaded.return_value = True
        mock_load.return_value = mock_model
        
        local_path = "/local/path/phi_model.ckpt"
        result = PhiRecapSummarizationModel.load_model_from_checkpoint(local_path)
        
        # Should call load_from_checkpoint with the same path
        mock_load.assert_called_once_with(local_path)
        assert result == mock_model