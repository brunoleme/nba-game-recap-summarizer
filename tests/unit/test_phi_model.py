import pytest
import torch
from unittest.mock import MagicMock, patch
from nba_game_recap_summarizer.finetuning.models.phi_model import PhiRecapSummarizationModel


class TestPhiRecapSummarizationModel:
    """Test cases for PhiRecapSummarizationModel."""

    def test_phi_model_initialization(self):
        """Test Phi model initialization with default parameters."""
        model = PhiRecapSummarizationModel(
            model_name="microsoft/Phi-3.5-mini-instruct",
            model_type="phi",
            use_quantization=False,
            peft_method="lora",
        )
        
        assert model.model_name == "microsoft/Phi-3.5-mini-instruct"
        assert model.model_type == "phi"
        assert model.use_quantization == False
        assert model.peft_method == "lora"

    def test_phi_model_with_quantization(self):
        """Test Phi model initialization with quantization."""
        model = PhiRecapSummarizationModel(
            model_name="microsoft/Phi-3.5-mini-instruct",
            model_type="phi",
            use_quantization=True,
            quantization_type="4bit",
            peft_method="lora",
        )
        
        assert model.use_quantization == True
        assert model.quantization_type == "4bit"

    @patch('nba_game_recap_summarizer.finetuning.models.phi_model.AutoTokenizer.from_pretrained')
    @patch('nba_game_recap_summarizer.finetuning.models.phi_model.AutoModelForCausalLM.from_pretrained')
    def test_phi_model_initialization_with_mocks(self, mock_model, mock_tokenizer):
        """Test Phi model initialization with mocked dependencies."""
        # Mock tokenizer
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"
        
        # Mock model
        mock_model.return_value = MagicMock()
        
        model = PhiRecapSummarizationModel(
            model_name="microsoft/Phi-3.5-mini-instruct",
            model_type="phi",
            use_quantization=False,
            peft_method="lora",
        )
        
        # Verify tokenizer was called with correct parameters
        mock_tokenizer.assert_called_once_with(
            "microsoft/Phi-3.5-mini-instruct", 
            use_fast=False, 
            trust_remote_code=True
        )
        
        # Verify model was called with correct parameters
        mock_model.assert_called_once()

    def test_phi_model_forward(self):
        """Test Phi model forward pass."""
        model = PhiRecapSummarizationModel(
            model_name="microsoft/Phi-3.5-mini-instruct",
            model_type="phi",
            use_quantization=False,
            peft_method="lora",
        )
        
        # Mock the model's forward method
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(0.5)
        model.model = MagicMock()
        model.model.return_value = mock_output
        
        # Test forward pass
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[1, 2, 3]])
        }
        
        output = model.forward(**inputs)
        assert output == mock_output

    @patch('nba_game_recap_summarizer.finetuning.models.phi_model.torch.load')
    def test_load_model_from_checkpoint_success(self, mock_torch_load):
        """Test successful model loading from checkpoint."""
        # Mock checkpoint data
        mock_checkpoint_data = {
            "hyper_parameters": {
                "model_name": "microsoft/Phi-3.5-mini-instruct",
                "model_type": "phi",
                "use_quantization": True,
                "quantization_type": "4bit",
                "peft_method": "lora",
            },
            "state_dict": {}
        }
        mock_torch_load.return_value = mock_checkpoint_data
        
        # Mock the model class
        with patch('nba_game_recap_summarizer.finetuning.models.phi_model.PhiRecapSummarizationModel.load_from_checkpoint') as mock_load_from_checkpoint:
            mock_model = MagicMock()
            mock_load_from_checkpoint.return_value = mock_model
            
            result = PhiRecapSummarizationModel.load_model_from_checkpoint(
                checkpoint_path="/path/to/checkpoint.ckpt"
            )
            
            assert result == mock_model
            mock_load_from_checkpoint.assert_called_once_with("/path/to/checkpoint.ckpt")

    @patch('nba_game_recap_summarizer.finetuning.models.phi_model.torch.load')
    def test_load_model_from_checkpoint_with_fallback(self, mock_torch_load):
        """Test model loading with quantization metadata mismatch fallback."""
        # Mock checkpoint data with quantization metadata
        mock_checkpoint_data = {
            "hyper_parameters": {
                "model_name": "microsoft/Phi-3.5-mini-instruct",
                "model_type": "phi",
                "use_quantization": True,
                "quantization_type": "4bit",
                "peft_method": "lora",
            },
            "state_dict": {}
        }
        mock_torch_load.return_value = mock_checkpoint_data
        
        # Mock the model class
        with patch('nba_game_recap_summarizer.finetuning.models.phi_model.PhiRecapSummarizationModel.load_from_checkpoint') as mock_load_from_checkpoint:
            # First call fails with quantization metadata mismatch
            mock_load_from_checkpoint.side_effect = Exception("Unexpected key(s) in state_dict: absmax")
            
            # Mock the model instantiation
            with patch('nba_game_recap_summarizer.finetuning.models.phi_model.PhiRecapSummarizationModel.__init__') as mock_init:
                mock_init.return_value = None
                
                result = PhiRecapSummarizationModel.load_model_from_checkpoint(
                    checkpoint_path="/path/to/checkpoint.ckpt"
                )
                
                # Should have called torch.load for fallback
                mock_torch_load.assert_called_once_with("/path/to/checkpoint.ckpt", map_location="cpu")

    def test_is_loaded(self):
        """Test model loaded status check."""
        model = PhiRecapSummarizationModel(
            model_name="microsoft/Phi-3.5-mini-instruct",
            model_type="phi",
            use_quantization=False,
            peft_method="lora",
        )
        
        # Test when model and tokenizer are None
        model.model = None
        model.tokenizer = None
        assert not model.is_loaded()
        
        # Test when model and tokenizer are present
        model.model = MagicMock()
        model.tokenizer = MagicMock()
        assert model.is_loaded()

    def test_summarize_recap(self):
        """Test single recap summarization."""
        model = PhiRecapSummarizationModel(
            model_name="microsoft/Phi-3.5-mini-instruct",
            model_type="phi",
            use_quantization=False,
            peft_method="lora",
        )
        
        # Mock tokenizer and model
        model.tokenizer = MagicMock()
        model.tokenizer.apply_chat_template.return_value = "System: You are an NBA Analyst. User: ### NBA Game Recap ###\nTest recap\n\n### Recap Summary ###"
        model.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
        }
        model.tokenizer.model_max_length = 2048
        model.tokenizer.eos_token_id = 2
        model.tokenizer.pad_token_id = 0
        
        model.model = MagicMock()
        model.model.eval.return_value = None
        model.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        model.device = torch.device("cpu")
        
        # Mock torch.no_grad
        with patch('torch.no_grad'):
            result = model.summarize_recap("Test NBA game recap")
            
            # Verify tokenizer was called
            model.tokenizer.apply_chat_template.assert_called_once()
            model.tokenizer.assert_called_once()
            
            # Verify model was called
            model.model.eval.assert_called_once()
            model.model.generate.assert_called_once()
            
            # Verify result
            assert isinstance(result, str)
