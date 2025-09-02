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

@patch('nba_game_recap_summarizer.finetuning.models.llama_model.boto3.client')
@patch('nba_game_recap_summarizer.finetuning.models.llama_model.tempfile.NamedTemporaryFile')
@patch('nba_game_recap_summarizer.finetuning.models.llama_model.Path')
@patch('nba_game_recap_summarizer.finetuning.models.llama_model.LlamaRecapSummarizationModel.load_from_checkpoint')
def test_load_model_from_checkpoint_s3(mock_load_from_checkpoint, mock_path, mock_tempfile, mock_boto3_client):
    """Test loading model from S3 checkpoint."""
    # Setup mocks
    mock_s3_client = MagicMock()
    mock_boto3_client.return_value = mock_s3_client
    
    # Mock file size response
    mock_s3_client.head_object.return_value = {'ContentLength': 1024 * 1024 * 100}  # 100MB
    
    # Mock temporary file
    mock_temp_file = MagicMock()
    mock_temp_file.name = '/tmp/test_model.ckpt'
    mock_tempfile.return_value = mock_temp_file
    
    # Mock Path operations
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = True
    mock_path_instance.stat.return_value.st_size = 1024 * 1024 * 100  # 100MB
    mock_path.return_value = mock_path_instance
    
    # Mock successful model loading
    mock_model = MagicMock()
    mock_model.is_loaded.return_value = True
    mock_load_from_checkpoint.return_value = mock_model
    
    # Test S3 path
    s3_path = "s3://test-bucket/models/test_model.ckpt"
    
    # Call the method
    result = LlamaRecapSummarizationModel.load_model_from_checkpoint(s3_path)
    
    # Verify S3 client was called correctly
    mock_s3_client.head_object.assert_called_once_with(Bucket='test-bucket', Key='models/test_model.ckpt')
    mock_s3_client.download_file.assert_called_once()
    
    # Verify model was loaded from local path
    mock_load_from_checkpoint.assert_called_once()
    
    # Verify result
    assert result == mock_model

def test_load_model_from_checkpoint_local():
    """Test loading model from local checkpoint."""
    with patch('nba_game_recap_summarizer.finetuning.models.llama_model.LlamaRecapSummarizationModel.load_from_checkpoint') as mock_load:
        mock_model = MagicMock()
        mock_model.is_loaded.return_value = True
        mock_load.return_value = mock_model
        
        local_path = "/local/path/model.ckpt"
        result = LlamaRecapSummarizationModel.load_model_from_checkpoint(local_path)
        
        # Should call load_from_checkpoint with the same path
        mock_load.assert_called_once_with(local_path)
        assert result == mock_model
