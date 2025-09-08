import pytest
from unittest.mock import patch, MagicMock
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
    with patch('nba_game_recap_summarizer.finetuning.models.llama_model.torch.load') as mock_torch_load, \
         patch('nba_game_recap_summarizer.finetuning.models.llama_model.LlamaRecapSummarizationModel') as mock_model_class:
        
        # Mock the torch.load to return a checkpoint with hyperparameters
        mock_checkpoint = {
            'hyper_parameters': {
                'model_name': 'meta-llama/Llama-3.2-1B-Instruct',
                'model_type': 'llama',
                'use_quantization': True,
                'quantization_type': '4bit',
                'peft_method': 'lora'
            },
            'state_dict': {
                'model.base_model.model.model.layers.0.self_attn.q_proj.weight': 'mock_weight',
                'model.base_model.model.model.layers.0.self_attn.k_proj.weight': 'mock_weight'
            }
        }
        mock_torch_load.return_value = mock_checkpoint
        
        # Mock the model instance
        mock_model = MagicMock()
        mock_model.is_loaded.return_value = True
        mock_model_class.return_value = mock_model
        
        local_path = "/local/path/model.ckpt"
        result = LlamaRecapSummarizationModel.load_model_from_checkpoint(local_path)
        
        # Should call torch.load first to get hyperparameters
        mock_torch_load.assert_called_once_with(local_path, map_location="cpu")
        
        # Should create a new model instance with the extracted parameters
        mock_model_class.assert_called_once_with(
            model_name='meta-llama/Llama-3.2-1B-Instruct',
            model_type='llama',
            use_quantization=True,
            quantization_type='4bit',
            peft_method='lora'
        )
        
        # Should load the state dict
        mock_model.load_state_dict.assert_called_once_with(mock_checkpoint['state_dict'], strict=False)
        
        assert result == mock_model
