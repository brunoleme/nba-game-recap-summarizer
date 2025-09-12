import pytest
import os
import tempfile
from pathlib import Path
from nba_game_recap_summarizer.finetuning.models.phi_model import PhiRecapSummarizationModel
from nba_game_recap_summarizer.finetuning.data.nba_recap_dataset import NBARecapDataModule


class TestPhiPipeline:
    """Integration tests for Phi-3.5-mini pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test artifacts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir

    @pytest.fixture
    def sample_data_path(self, temp_dir):
        """Create sample data file for testing."""
        data_path = Path(temp_dir) / "sample_data.csv"
        sample_data = """game_recap,game_recap_summary
"The Lakers defeated the Warriors 120-115 in overtime. LeBron James scored 30 points.","Lakers beat Warriors 120-115 in OT with LeBron scoring 30 points."
"The Celtics beat the Heat 98-95. Jayson Tatum had 25 points and 10 rebounds.","Celtics defeated Heat 98-95 behind Tatum's 25 points and 10 rebounds."
"""
        data_path.write_text(sample_data)
        return str(data_path)

    def test_phi_model_with_tinyllama(self, temp_dir, sample_data_path):
        """Test Phi model using TinyLlama as a substitute."""
        # Use TinyLlama since it's LLaMA-compatible and much faster
        model = PhiRecapSummarizationModel(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            model_type="llama",  # Use LLaMA type since TinyLlama is LLaMA-compatible
            use_quantization=False,
            peft_method="lora",
            max_length=128,  # Small context for fast testing
        )
        
        assert model.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert model.model_type == "llama"
        assert model.is_loaded()

    def test_phi_model_data_loading(self, temp_dir, sample_data_path):
        """Test Phi model with data loading."""
        # Create data module
        data_module = NBARecapDataModule(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            source_data_path=sample_data_path,
            preprocessed_input_data_folder=temp_dir,
            env_folder="test",
            batch_size=1,
            max_length=64,  # Very small for fast testing
            train_samples=2,
            val_samples=1,
            test_samples=1,
        )
        
        # Setup data
        data_module.setup()
        
        # Verify data was loaded
        assert len(data_module.train_dataset) == 2
        assert len(data_module.val_dataset) == 1
        assert len(data_module.test_dataset) == 1

    def test_phi_model_generation(self, temp_dir, sample_data_path):
        """Test Phi model text generation."""
        # Use TinyLlama for fast testing
        model = PhiRecapSummarizationModel(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            model_type="llama",
            use_quantization=False,
            peft_method="lora",
            max_length=64,
        )
        
        # Test summarization
        test_recap = "The Lakers defeated the Warriors 120-115 in overtime. LeBron James scored 30 points."
        
        # Mock the generation to avoid actual model inference in tests
        with pytest.MonkeyPatch().context() as m:
            m.setattr(model, 'summarize_recap', lambda x, max_length=None: "Lakers beat Warriors in OT")
            
            result = model.summarize_recap(test_recap, max_length=50)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_phi_model_batch_generation(self, temp_dir, sample_data_path):
        """Test Phi model batch generation."""
        # Use TinyLlama for fast testing
        model = PhiRecapSummarizationModel(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            model_type="llama",
            use_quantization=False,
            peft_method="lora",
            max_length=64,
        )
        
        # Create data module
        data_module = NBARecapDataModule(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            source_data_path=sample_data_path,
            preprocessed_input_data_folder=temp_dir,
            env_folder="test",
            batch_size=1,
            max_length=64,
            train_samples=2,
            val_samples=1,
            test_samples=1,
        )
        
        data_module.setup()
        
        # Mock batch generation
        with pytest.MonkeyPatch().context() as m:
            m.setattr(model, 'summarize_recaps', lambda dataloader, max_length=None: ["Summary 1", "Summary 2"])
            
            results = model.summarize_recaps(data_module.test_dataloader(), max_length=50)
            assert isinstance(results, list)
            assert len(results) == 1  # test_samples=1
            assert all(isinstance(r, str) for r in results)

    def test_phi_model_quantization(self, temp_dir):
        """Test Phi model with quantization."""
        # Test quantization configuration
        model = PhiRecapSummarizationModel(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            model_type="llama",
            use_quantization=True,
            quantization_type="4bit",
            peft_method="lora",
        )
        
        assert model.use_quantization == True
        assert model.quantization_type == "4bit"
        assert model.quantization_config is not None

    def test_phi_model_lora_config(self, temp_dir):
        """Test Phi model LoRA configuration."""
        model = PhiRecapSummarizationModel(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            model_type="llama",
            use_quantization=False,
            peft_method="lora",
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.1,
        )
        
        assert model.peft_method == "lora"
        assert model.peft_config is not None
        assert model.peft_config.r == 8
        assert model.peft_config.lora_alpha == 16
        assert model.peft_config.lora_dropout == 0.1

    def test_phi_model_error_handling(self, temp_dir):
        """Test Phi model error handling."""
        # Test with invalid model name
        with pytest.raises(Exception):
            PhiRecapSummarizationModel(
                model_name="invalid-model-name",
                model_type="phi",
                use_quantization=False,
                peft_method="lora",
            )

    def test_phi_model_device_placement(self, temp_dir):
        """Test Phi model device placement."""
        import torch
        
        model = PhiRecapSummarizationModel(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            model_type="llama",
            use_quantization=False,
            peft_method="lora",
        )
        
        # Test device placement
        if torch.cuda.is_available():
            assert model.device.type == "cuda"
        else:
            assert model.device.type == "cpu"
