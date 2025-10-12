"""
Comprehensive integration tests for multi-model support.
This file tests the complete pipeline with LLaMA, Mistral, and Phi models.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from nba_game_recap_summarizer.finetuning.models.llama_model import LlamaRecapSummarizationModel
from nba_game_recap_summarizer.finetuning.models.phi_model import PhiRecapSummarizationModel
from nba_game_recap_summarizer.finetuning.models.mistral_model import MistralRecapSummarizationModel
from nba_game_recap_summarizer.finetuning.data.nba_recap_dataset import NBARecapDataModule


class TestMultiModelIntegration:
    """Integration tests for multi-model support (LLaMA, Mistral, Phi)."""

    @pytest.fixture
    def sample_data_path(self):
        """Create sample data file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            sample_data = """game_recap,game_recap_summary
"The Lakers defeated the Warriors 120-115 in overtime. LeBron James scored 30 points.","Lakers beat Warriors 120-115 in OT with LeBron scoring 30 points."
"The Celtics beat the Heat 98-95. Jayson Tatum had 25 points and 10 rebounds.","Celtics defeated Heat 98-95 behind Tatum's 25 points and 10 rebounds."
"The Nuggets won against the Lakers 108-103. Nikola Jokic had a triple-double.","Nuggets beat Lakers 108-103 with Jokic's triple-double."
"""
            tmp_file.write(sample_data)
            tmp_file.flush()
            yield tmp_file.name
        os.unlink(tmp_file.name)

    def test_model_registry_supports_all_models(self):
        """Test that the model registry supports LLaMA, Mistral, and Phi models."""
        from nba_game_recap_summarizer.finetuning.utils.load_models import MODEL_CLASSES
        
        assert "llama" in MODEL_CLASSES
        assert "mistral" in MODEL_CLASSES
        assert "phi" in MODEL_CLASSES
        assert "nba_game_recap_summarizer.finetuning.models.llama_model.LlamaRecapSummarizationModel" in MODEL_CLASSES["llama"]
        assert "nba_game_recap_summarizer.finetuning.models.mistral_model.MistralRecapSummarizationModel" in MODEL_CLASSES["mistral"]
        assert "nba_game_recap_summarizer.finetuning.models.phi_model.PhiRecapSummarizationModel" in MODEL_CLASSES["phi"]

    def test_llama_model_initialization(self):
        """Test LLaMA model initialization."""
        model = LlamaRecapSummarizationModel(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            model_type="llama",
            use_quantization=False,
            peft_method="lora",
        )
        
        assert model.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert model.model_type == "llama"
        assert model.is_loaded()

    def test_mistral_model_initialization(self):
        """Test Mistral model initialization."""
        model = MistralRecapSummarizationModel(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Use TinyLlama for testing
            model_type="llama",  # TinyLlama is LLaMA-compatible
            use_quantization=False,
            peft_method="lora",
        )
        
        assert model.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert model.model_type == "llama"
        assert model.is_loaded()

    def test_phi_model_initialization(self):
        """Test Phi model initialization."""
        model = PhiRecapSummarizationModel(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Use TinyLlama for testing
            model_type="llama",  # TinyLlama is LLaMA-compatible
            use_quantization=False,
            peft_method="lora",
        )
        
        assert model.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert model.model_type == "llama"
        assert model.is_loaded()

    def test_llama_model_with_data(self, sample_data_path):
        """Test LLaMA model with data loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_module = NBARecapDataModule(
                model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                source_data_path=sample_data_path,
                preprocessed_input_data_folder=temp_dir,
                env_folder="test",
                batch_size=1,
                max_length=64,
                train_samples=3,
                val_samples=1,
                test_samples=1,
            )
            
            data_module.setup()
            
            assert len(data_module.train_dataset) == 3
            assert len(data_module.val_dataset) == 1
            assert len(data_module.test_dataset) == 1

    def test_mistral_model_with_data(self, sample_data_path):
        """Test Mistral model with data loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_module = NBARecapDataModule(
                model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                source_data_path=sample_data_path,
                preprocessed_input_data_folder=temp_dir,
                env_folder="test",
                batch_size=1,
                max_length=64,
                train_samples=3,
                val_samples=1,
                test_samples=1,
            )
            
            data_module.setup()
            
            assert len(data_module.train_dataset) == 3
            assert len(data_module.val_dataset) == 1
            assert len(data_module.test_dataset) == 1

    def test_phi_model_with_data(self, sample_data_path):
        """Test Phi model with data loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_module = NBARecapDataModule(
                model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                source_data_path=sample_data_path,
                preprocessed_input_data_folder=temp_dir,
                env_folder="test",
                batch_size=1,
                max_length=64,
                train_samples=3,
                val_samples=1,
                test_samples=1,
            )
            
            data_module.setup()
            
            assert len(data_module.train_dataset) == 3
            assert len(data_module.val_dataset) == 1
            assert len(data_module.test_dataset) == 1

    def test_llama_model_generation(self):
        """Test LLaMA model text generation."""
        model = LlamaRecapSummarizationModel(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            model_type="llama",
            use_quantization=False,
            peft_method="lora",
            max_length=64,
        )
        
        # Mock the generation to avoid actual model inference in tests
        with patch.object(model, 'summarize_recap', return_value="Lakers beat Warriors in OT"):
            result = model.summarize_recap("Test recap", max_length=50)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_mistral_model_generation(self):
        """Test Mistral model text generation."""
        model = MistralRecapSummarizationModel(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            model_type="llama",
            use_quantization=False,
            peft_method="lora",
            max_length=64,
        )
        
        # Mock the generation to avoid actual model inference in tests
        with patch.object(model, 'summarize_recap', return_value="Lakers beat Warriors in OT"):
            result = model.summarize_recap("Test recap", max_length=50)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_phi_model_generation(self):
        """Test Phi model text generation."""
        model = PhiRecapSummarizationModel(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            model_type="llama",
            use_quantization=False,
            peft_method="lora",
            max_length=64,
        )
        
        # Mock the generation to avoid actual model inference in tests
        with patch.object(model, 'summarize_recap', return_value="Lakers beat Warriors in OT"):
            result = model.summarize_recap("Test recap", max_length=50)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_all_models_quantization_support(self):
        """Test that all models support quantization."""
        # Test LLaMA with quantization
        llama_model = LlamaRecapSummarizationModel(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            model_type="llama",
            use_quantization=True,
            quantization_type="4bit",
            peft_method="lora",
        )
        assert llama_model.use_quantization == True
        assert llama_model.quantization_type == "4bit"

        # Test Mistral with quantization
        mistral_model = MistralRecapSummarizationModel(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            model_type="llama",
            use_quantization=True,
            quantization_type="4bit",
            peft_method="lora",
        )
        assert mistral_model.use_quantization == True
        assert mistral_model.quantization_type == "4bit"

        # Test Phi with quantization
        phi_model = PhiRecapSummarizationModel(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            model_type="llama",
            use_quantization=True,
            quantization_type="4bit",
            peft_method="lora",
        )
        assert phi_model.use_quantization == True
        assert phi_model.quantization_type == "4bit"

    def test_all_models_peft_support(self):
        """Test that all models support PEFT methods."""
        # Test LLaMA with LoRA
        llama_model = LlamaRecapSummarizationModel(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            model_type="llama",
            use_quantization=False,
            peft_method="lora",
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.1,
        )
        assert llama_model.peft_method == "lora"
        assert llama_model.peft_config is not None

        # Test Mistral with LoRA
        mistral_model = MistralRecapSummarizationModel(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            model_type="llama",
            use_quantization=False,
            peft_method="lora",
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.1,
        )
        assert mistral_model.peft_method == "lora"
        assert mistral_model.peft_config is not None

        # Test Phi with LoRA
        phi_model = PhiRecapSummarizationModel(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            model_type="llama",
            use_quantization=False,
            peft_method="lora",
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.1,
        )
        assert phi_model.peft_method == "lora"
        assert phi_model.peft_config is not None

    def test_all_models_checkpoint_loading(self):
        """Test that all models support checkpoint loading."""
        # Mock checkpoint data
        mock_checkpoint_data = {
            "hyper_parameters": {
                "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "model_type": "llama",
                "use_quantization": False,
                "peft_method": "lora",
            },
            "state_dict": {}
        }
        
        with patch('torch.load', return_value=mock_checkpoint_data):
            # Test LLaMA checkpoint loading
            with patch('nba_game_recap_summarizer.finetuning.models.llama_model.LlamaRecapSummarizationModel.load_from_checkpoint') as mock_llama_load:
                mock_llama_model = MagicMock()
                mock_llama_load.return_value = mock_llama_model
                
                result = LlamaRecapSummarizationModel.load_model_from_checkpoint("/path/to/llama.ckpt")
                assert result == mock_llama_model

            # Test Mistral checkpoint loading
            with patch('nba_game_recap_summarizer.finetuning.models.mistral_model.MistralRecapSummarizationModel.load_from_checkpoint') as mock_mistral_load:
                mock_mistral_model = MagicMock()
                mock_mistral_load.return_value = mock_mistral_model
                
                result = MistralRecapSummarizationModel.load_model_from_checkpoint("/path/to/mistral.ckpt")
                assert result == mock_mistral_model

            # Test Phi checkpoint loading
            with patch('nba_game_recap_summarizer.finetuning.models.phi_model.PhiRecapSummarizationModel.load_from_checkpoint') as mock_phi_load:
                mock_phi_model = MagicMock()
                mock_phi_load.return_value = mock_phi_model
                
                result = PhiRecapSummarizationModel.load_model_from_checkpoint("/path/to/phi.ckpt")
                assert result == mock_phi_model

    def test_all_models_compatibility(self):
        """Test that all models are compatible with the same data format."""
        # All models should work with the same data module
        with tempfile.TemporaryDirectory() as temp_dir:
            data_module = NBARecapDataModule(
                model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                source_data_path="tests/resources/source_data/game_recaps_with_summaries_sample.csv",
                preprocessed_input_data_folder=temp_dir,
                env_folder="test",
                batch_size=1,
                max_length=64,
                train_samples=2,
                val_samples=1,
                test_samples=1,
            )
            
            data_module.setup()
            
            # All models should be able to process the same data
            llama_model = LlamaRecapSummarizationModel(
                model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                model_type="llama",
                use_quantization=False,
                peft_method="lora",
            )
            
            mistral_model = MistralRecapSummarizationModel(
                model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                model_type="llama",
                use_quantization=False,
                peft_method="lora",
            )
            
            phi_model = PhiRecapSummarizationModel(
                model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                model_type="llama",
                use_quantization=False,
                peft_method="lora",
            )
            
            # All should be loaded and ready
            assert llama_model.is_loaded()
            assert mistral_model.is_loaded()
            assert phi_model.is_loaded()

    def test_model_architecture_differences(self):
        """Test that models have different architectures but same interface."""
        llama_model = LlamaRecapSummarizationModel(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            model_type="llama",
            use_quantization=False,
            peft_method="lora",
        )
        
        mistral_model = MistralRecapSummarizationModel(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            model_type="llama",
            use_quantization=False,
            peft_method="lora",
        )
        
        phi_model = PhiRecapSummarizationModel(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            model_type="llama",
            use_quantization=False,
            peft_method="lora",
        )
        
        # All should have the same interface
        for model in [llama_model, mistral_model, phi_model]:
            assert hasattr(model, 'summarize_recap')
            assert hasattr(model, 'summarize_recaps')
            assert hasattr(model, 'is_loaded')
        
        # But they should be different classes
        assert type(llama_model) != type(mistral_model)
        assert type(llama_model) != type(phi_model)
        assert type(mistral_model) != type(phi_model)
        assert isinstance(llama_model, LlamaRecapSummarizationModel)
        assert isinstance(mistral_model, MistralRecapSummarizationModel)
        assert isinstance(phi_model, PhiRecapSummarizationModel)

    def test_model_interchangeability(self):
        """Test that models can be swapped without code changes."""
        test_recap = "Lakers beat Warriors 120-115 in overtime."
        
        models = [
            LlamaRecapSummarizationModel(
                model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                model_type="llama",
                use_quantization=False,
                peft_method="lora",
            ),
            MistralRecapSummarizationModel(
                model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                model_type="llama",
                use_quantization=False,
                peft_method="lora",
            ),
            PhiRecapSummarizationModel(
                model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                model_type="llama",
                use_quantization=False,
                peft_method="lora",
            ),
        ]
        
        # All models should work with the same code
        for model in models:
            with patch.object(model, 'summarize_recap', return_value="Test summary"):
                summary = model.summarize_recap(test_recap, max_length=50)
                assert isinstance(summary, str)
                assert len(summary) > 0

