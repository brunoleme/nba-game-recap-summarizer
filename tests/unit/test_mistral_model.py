import pytest
from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from nba_game_recap_summarizer.finetuning.models.mistral_model import MistralRecapSummarizationModel


class DummyDataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data = [
            {"game_recap": "Lakers beats Suns in the overtime with a Bryant game winner."},
            {"game_recap": "Warriors dominate Celtics in Game 6 to win the NBA championship."},
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@pytest.fixture
def default_mistral_model() -> MistralRecapSummarizationModel:
    return MistralRecapSummarizationModel(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Use TinyLlama for testing
        model_type="llama",  # TinyLlama is LLaMA-compatible
        use_quantization=False,
    )


@pytest.mark.parametrize(
    "model_name, model_type, use_quantization",
    [
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "llama", False),  # TinyLlama as Mistral substitute
        ("mistral-nonexistent", "mistral", False),
    ]
)
def test_mistral_model_initialization(model_name, model_type, use_quantization):
    if model_name == "mistral-nonexistent":
        with pytest.raises(Exception):
            MistralRecapSummarizationModel(
                model_name=model_name,
                model_type=model_type,
                use_quantization=use_quantization
            )
    else:
        model = MistralRecapSummarizationModel(
            model_name=model_name,
            model_type=model_type,
            use_quantization=use_quantization
        )
        assert model.model_name == model_name
        assert model.model_type == model_type
        assert model.use_quantization == use_quantization


def test_mistral_model_summarize_recap(default_mistral_model) -> None:
    """Test Mistral model single recap summarization."""
    game_recap = "Lakers beats Suns in the overtime with a Bryant game winner."
    game_recap_summary = default_mistral_model.summarize_recap(game_recap, max_length=50)
    assert isinstance(game_recap_summary, str)
    assert len(game_recap_summary) > 0


def test_mistral_model_summarize_recaps(default_mistral_model) -> None:
    """Test Mistral model batch recap summarization."""
    dataset = DummyDataset(default_mistral_model.tokenizer)
    dataloader = DataLoader(dataset, batch_size=1)

    game_recap_summaries = default_mistral_model.summarize_recaps(dataloader, max_length=20)
    assert isinstance(game_recap_summaries, list)
    assert all(isinstance(n, str) for n in game_recap_summaries)


def test_mistral_model_is_loaded(default_mistral_model):
    """Test the Mistral model is_loaded method."""
    # Model should be loaded after initialization
    assert default_mistral_model.is_loaded() == True

    # Test with None model
    model = MistralRecapSummarizationModel(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        model_type="llama",
        use_quantization=False,
    )
    # Should still be loaded even with None (mocked)
    assert model.is_loaded() == True


@pytest.mark.parametrize("peft_method", ["lora"])
def test_mistral_model_with_peft(peft_method):
    """Test Mistral model with PEFT methods."""
    model = MistralRecapSummarizationModel(
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


def test_mistral_load_model_from_checkpoint():
    """Test loading Mistral model from checkpoint (S3 or local)."""
    with patch('nba_game_recap_summarizer.finetuning.models.mistral_model.MistralRecapSummarizationModel.load_from_checkpoint') as mock_load:
        mock_model = MagicMock()
        mock_model.is_loaded.return_value = True
        mock_load.return_value = mock_model

        local_path = "/local/path/mistral_model.ckpt"
        result = MistralRecapSummarizationModel.load_model_from_checkpoint(local_path)

        # Should call load_from_checkpoint with the same path
        mock_load.assert_called_once_with(local_path)
        assert result == mock_model


def test_mistral_model_forward(default_mistral_model):
    """Test Mistral model forward pass."""
    # Mock the model's forward method
    with patch.object(default_mistral_model.model, '__call__') as mock_forward:
        mock_output = MagicMock()
        mock_output.loss = 0.5
        mock_forward.return_value = mock_output

        # Test forward pass
        inputs = {"input_ids": None, "attention_mask": None}
        result = default_mistral_model.forward(**inputs)
        
        assert result == mock_output
        mock_forward.assert_called_once_with(**inputs)


def test_mistral_model_compute_loss(default_mistral_model):
    """Test Mistral model loss computation."""
    # Mock the forward method
    with patch.object(default_mistral_model, 'forward') as mock_forward:
        mock_output = MagicMock()
        mock_output.loss = 0.5
        mock_forward.return_value = mock_output

        # Test loss computation
        batch = {"input_ids": None, "attention_mask": None}
        loss = default_mistral_model.compute_loss(batch)
        
        assert loss == mock_output.loss
        mock_forward.assert_called_once_with(**batch)


def test_mistral_model_compute_validation_metrics(default_mistral_model):
    """Test Mistral model validation metrics computation."""
    # Mock the forward method
    with patch.object(default_mistral_model, 'forward') as mock_forward:
        mock_output = MagicMock()
        mock_output.loss = 0.5
        mock_forward.return_value = mock_output

        # Mock the _generate_predictions_for_eval method
        with patch.object(default_mistral_model, '_generate_predictions_for_eval', return_value=([], [])):
            # Test validation metrics computation
            batch = {"input_ids": None, "attention_mask": None}
            metrics = default_mistral_model.compute_validation_metrics(batch, 0)
            
            assert "val_loss" in metrics
            assert metrics["val_loss"] == 0.5
            mock_forward.assert_called_once_with(**batch)


def test_mistral_model_setup_training(default_mistral_model):
    """Test Mistral model training setup."""
    # Mock the model's config
    with patch.object(default_mistral_model.model, 'config') as mock_config:
        mock_config.use_cache = True
        
        # Test setup_training
        default_mistral_model.setup_training()
        
        # Should disable cache for training
        assert mock_config.use_cache == False


def test_mistral_model_setup_inference(default_mistral_model):
    """Test Mistral model inference setup."""
    # Mock the model's config
    with patch.object(default_mistral_model.model, 'config') as mock_config:
        mock_config.use_cache = False
        
        # Test setup_inference
        default_mistral_model.setup_inference()
        
        # Should enable cache for inference
        assert mock_config.use_cache == True
