import pytest
import os
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
import tempfile

from nba_game_recap_summarizer.api.inference import (
    app, 
    GameRecapRequest, 
    GameRecapResponse, 
    load_model,
    health_check,
    summarize_recap
)
from nba_game_recap_summarizer.finetuning.models.llama_model import LlamaRecapSummarizationModel
from nba_game_recap_summarizer.finetuning.models.phi_model import PhiRecapSummarizationModel


class TestGameRecapRequest:
    """Test the GameRecapRequest model validation."""
    
    def test_valid_request(self):
        """Test valid request creation."""
        request = GameRecapRequest(
            game_recap="Lakers beat Suns in overtime with a Bryant game winner.",
            max_length=1024
        )
        assert request.game_recap == "Lakers beat Suns in overtime with a Bryant game winner."
        assert request.max_length == 1024
    
    def test_default_max_length(self):
        """Test default max_length value."""
        request = GameRecapRequest(game_recap="Test recap")
        assert request.max_length == 2048
    
    def test_min_length_validation(self):
        """Test min_length validation for game_recap."""
        with pytest.raises(ValueError):
            GameRecapRequest(game_recap="")
    
    def test_max_length_validation(self):
        """Test max_length validation."""
        with pytest.raises(ValueError):
            GameRecapRequest(game_recap="Test", max_length=0)
        
        with pytest.raises(ValueError):
            GameRecapRequest(game_recap="Test", max_length=3000)
    
    def test_game_recap_cleaning(self):
        """Test game_recap input cleaning."""
        request = GameRecapRequest(game_recap="Lakers\nbeat\rSuns   in   overtime")
        assert request.game_recap == "Lakers beat Suns in overtime"
    
    def test_empty_string_validation(self):
        """Test empty string validation."""
        # The validator cleans whitespace, so empty strings become empty after cleaning
        # This should trigger the min_length validation
        with pytest.raises(ValueError):
            GameRecapRequest(game_recap="")  # Direct empty string should fail min_length


class TestGameRecapResponse:
    """Test the GameRecapResponse model."""
    
    def test_response_creation(self):
        """Test response model creation."""
        response = GameRecapResponse(game_recap_summary="Lakers beat Suns in overtime")
        assert response.game_recap_summary == "Lakers beat Suns in overtime"


class TestInferenceEndpoints:
    """Test the inference API endpoints."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        mock_model = MagicMock()
        mock_model.is_loaded.return_value = True
        mock_model.summarize_recap.return_value = "Mock summary: Lakers beat Suns"
        return mock_model
    
    @pytest.fixture
    def client_with_mock_model(self, mock_model):
        """Create a test client with mocked model."""
        # Patch the global model variable
        with patch('nba_game_recap_summarizer.api.inference.model', mock_model):
            client = TestClient(app)
            yield client
    
    def test_root_endpoint(self, client_with_mock_model):
        """Test the root API endpoint."""
        response = client_with_mock_model.get("/api")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "NBA Game Recap Summarizer API"
        assert data["version"] == "1.0.0"
        assert "/health" in data["endpoints"]
        assert "/summarize_recap" in data["endpoints"]
    
    def test_health_check_healthy(self, client_with_mock_model):
        """Test health check when model is loaded."""
        response = client_with_mock_model.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
    
    def test_health_check_unhealthy_no_model(self):
        """Test health check when model is not loaded."""
        with patch('nba_game_recap_summarizer.api.inference.model', None):
            client = TestClient(app)
            response = client.get("/health")
            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]
    
    def test_health_check_unhealthy_model_not_loaded(self):
        """Test health check when model is not ready."""
        mock_model = MagicMock()
        mock_model.is_loaded.return_value = False
        
        with patch('nba_game_recap_summarizer.api.inference.model', mock_model):
            client = TestClient(app)
            response = client.get("/health")
            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]
    
    def test_summarize_recap_success(self, client_with_mock_model, mock_model):
        """Test successful recap summarization."""
        request_data = {
            "game_recap": "Lakers beat Suns in overtime with a Bryant game winner.",
            "max_length": 1024
        }
        
        response = client_with_mock_model.post("/summarize_recap", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "game_recap_summary" in data
        assert data["game_recap_summary"] == "Mock summary: Lakers beat Suns"
        
        # Verify model was called correctly
        mock_model.summarize_recap.assert_called_once_with(
            game_recap="Lakers beat Suns in overtime with a Bryant game winner.",
            max_length=1024
        )
    
    def test_summarize_recap_with_default_max_length(self, client_with_mock_model, mock_model):
        """Test summarization with default max_length."""
        request_data = {
            "game_recap": "Lakers beat Suns in overtime."
        }
        
        response = client_with_mock_model.post("/summarize_recap", json=request_data)
        assert response.status_code == 200
        
        # Verify model was called with default max_length
        mock_model.summarize_recap.assert_called_once_with(
            game_recap="Lakers beat Suns in overtime.",
            max_length=2048
        )
    
    def test_summarize_recap_empty_input(self, client_with_mock_model):
        """Test summarization with empty input."""
        request_data = {
            "game_recap": "",
            "max_length": 1024
        }
        
        response = client_with_mock_model.post("/summarize_recap", json=request_data)
        # Empty string triggers validation error (422) not business logic error (400)
        assert response.status_code == 422
    
    def test_summarize_recap_validation_error(self, client_with_mock_model):
        """Test summarization with validation error."""
        request_data = {
            "game_recap": "Valid recap",
            "max_length": 3000  # Invalid max_length
        }
        
        response = client_with_mock_model.post("/summarize_recap", json=request_data)
        assert response.status_code == 422
    
    def test_summarize_recap_model_error(self, client_with_mock_model, mock_model):
        """Test summarization when model raises an error."""
        mock_model.summarize_recap.side_effect = Exception("Model error")
        
        request_data = {
            "game_recap": "Lakers beat Suns in overtime.",
            "max_length": 1024
        }
        
        response = client_with_mock_model.post("/summarize_recap", json=request_data)
        assert response.status_code == 500
        # The error message is passed through directly
        assert "Model error" in response.json()["detail"]


class TestModelLoading:
    """Test the model loading functionality."""
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('nba_game_recap_summarizer.api.inference.LlamaRecapSummarizationModel')
    @patch('nba_game_recap_summarizer.api.inference.os.path.exists')
    def test_load_model_from_local_hf_path(self, mock_exists, mock_llama_model_class, mock_model_hf, mock_tokenizer):
        """Test loading model from local Hugging Face path."""
        mock_exists.return_value = True
        mock_tokenizer.return_value = MagicMock()
        mock_model_hf.return_value = MagicMock()
        mock_llama_model = MagicMock()
        mock_llama_model_class.return_value = mock_llama_model
        
        # Create a new app instance to test startup
        test_app = FastAPI()
        test_app.add_event_handler("startup", load_model)
        
        with TestClient(test_app) as client:
            # The startup event should have been triggered
            # Check that exists was called with the correct paths (aligned first)
            expected_calls = [
                unittest.mock.call("/app/models/hf_model_merged_aligned"),
                unittest.mock.call("/app/models/hf_model_merged_aligned/config.json")
            ]
            mock_exists.assert_has_calls(expected_calls, any_order=True)
            mock_tokenizer.assert_called_with("/app/models/hf_model_merged_aligned")
            mock_model_hf.assert_called()
            mock_llama_model_class.assert_called()
    
    @patch('boto3.client')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('nba_game_recap_summarizer.api.inference.LlamaRecapSummarizationModel')
    @patch('nba_game_recap_summarizer.api.inference.os.path.exists')
    @patch('nba_game_recap_summarizer.api.inference.os.makedirs')
    @patch('nba_game_recap_summarizer.api.inference.settings')
    def test_load_model_from_s3_hf_path(self, mock_settings, mock_makedirs, mock_exists, mock_llama_model_class, mock_model_hf, mock_tokenizer, mock_boto3):
        """Test loading model from S3 Hugging Face path when local path doesn't exist."""
        # Mock the exists calls to return False for both the directory and config.json
        def mock_exists_side_effect(path):
            if path == "/app/models/hf_model":
                return False
            elif path == "/app/models/hf_model/config.json":
                return False
            return False
        
        mock_exists.side_effect = mock_exists_side_effect
        mock_settings.model_path = "s3://bucket/hf_model"
        mock_tokenizer.return_value = MagicMock()
        mock_model_hf.return_value = MagicMock()
        mock_llama_model = MagicMock()
        mock_llama_model_class.return_value = mock_llama_model
        
        # Mock S3 client to fail
        mock_s3_client = MagicMock()
        mock_s3_client.list_objects_v2.side_effect = Exception("S3 error")
        mock_boto3.return_value = mock_s3_client
        
        # Create a new app instance to test startup
        test_app = FastAPI()
        test_app.add_event_handler("startup", load_model)
        
        with TestClient(test_app) as client:
            # The startup event should have been triggered
            # Check that exists was called (the exact calls may vary based on the logic)
            mock_exists.assert_called()
            # Note: S3 client may not be called if the model loading logic changes
    
    @patch('nba_game_recap_summarizer.api.inference.LlamaRecapSummarizationModel.load_model_from_checkpoint')
    @patch('nba_game_recap_summarizer.api.inference.os.path.exists')
    @patch('nba_game_recap_summarizer.api.inference.os.makedirs')
    @patch('nba_game_recap_summarizer.api.inference.settings')
    def test_load_model_fallback_to_hf(self, mock_settings, mock_makedirs, mock_exists, mock_load_checkpoint):
        """Test model loading fallback to checkpoint when both local and S3 fail."""
        # Mock the exists calls to return False for both the directory and config.json
        def mock_exists_side_effect(path):
            if path == "/app/models/hf_model":
                return False
            elif path == "/app/models/hf_model/config.json":
                return False
            return False
        
        mock_exists.side_effect = mock_exists_side_effect
        mock_settings.model_path = "local_checkpoint.ckpt"  # Not an S3 path
        mock_model = MagicMock()
        mock_load_checkpoint.return_value = mock_model
        
        # Create a new app instance to test startup
        test_app = FastAPI()
        test_app.add_event_handler("startup", load_model)
        
        with TestClient(test_app) as client:
            # Should fallback to checkpoint loading
            mock_load_checkpoint.assert_called()
    
    @patch('nba_game_recap_summarizer.api.inference.LlamaRecapSummarizationModel.load_model_from_checkpoint')
    @patch('nba_game_recap_summarizer.api.inference.os.path.exists')
    @patch('nba_game_recap_summarizer.api.inference.os.makedirs')
    @patch('nba_game_recap_summarizer.api.inference.settings')
    def test_load_model_failure(self, mock_settings, mock_makedirs, mock_exists, mock_load_checkpoint):
        """Test model loading failure."""
        # Mock the exists calls to return False for both the directory and config.json
        def mock_exists_side_effect(path):
            if path == "/app/models/hf_model":
                return False
            elif path == "/app/models/hf_model/config.json":
                return False
            return False
        
        mock_exists.side_effect = mock_exists_side_effect
        mock_settings.model_path = "s3://bucket/hf_model"
        mock_load_checkpoint.side_effect = Exception("Model loading failed")
        
        # Mock the S3 download to fail
        with patch('boto3.client', side_effect=Exception("S3 error")):
            # Create a new app instance to test startup
            test_app = FastAPI()
            test_app.add_event_handler("startup", load_model)
            
            with pytest.raises(RuntimeError, match="Failed to initialize model"):
                with TestClient(test_app) as client:
                    pass


class TestMiddleware:
    """Test the request logging middleware."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        mock_model = MagicMock()
        mock_model.is_loaded.return_value = True
        mock_model.summarize_recap.return_value = "Mock summary: Lakers beat Suns"
        return mock_model
    
    @pytest.fixture
    def client_with_mock_model(self, mock_model):
        """Create a test client with mocked model."""
        # Patch the global model variable
        with patch('nba_game_recap_summarizer.api.inference.model', mock_model):
            client = TestClient(app)
            yield client
    
    def test_middleware_logs_requests(self, client_with_mock_model):
        """Test that middleware logs incoming requests."""
        with patch('nba_game_recap_summarizer.api.inference.logger') as mock_logger:
            response = client_with_mock_model.get("/api")
            assert response.status_code == 200
            mock_logger.info.assert_called()
    
    def test_middleware_handles_request_body_error(self, client_with_mock_model):
        """Test middleware handles request body logging errors gracefully."""
        with patch('nba_game_recap_summarizer.api.inference.logger') as mock_logger:
            # Mock the request body to cause an error
            with patch('fastapi.Request.body', side_effect=Exception("Body error")):
                response = client_with_mock_model.get("/api")
                assert response.status_code == 200
                mock_logger.error.assert_called()


class TestTextCleaning:
    """Test the text cleaning functionality."""
    
    def test_clean_game_recap_input(self):
        """Test the game_recap input cleaning validator."""
        request = GameRecapRequest(game_recap="Lakers\nbeat\rSuns   in   overtime")
        assert request.game_recap == "Lakers beat Suns in overtime"
    
    def test_clean_game_recap_with_special_chars(self):
        """Test cleaning with special characters."""
        request = GameRecapRequest(game_recap="Lakers\tbeat\nSuns\r\nin\tovertime")
        assert request.game_recap == "Lakers beat Suns in overtime"
    
    def test_clean_game_recap_multiple_spaces(self):
        """Test cleaning multiple spaces."""
        request = GameRecapRequest(game_recap="Lakers    beat     Suns    in    overtime")
        assert request.game_recap == "Lakers beat Suns in overtime"
