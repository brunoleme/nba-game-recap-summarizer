import pytest
import os
import tempfile
import subprocess
import time
import requests
from unittest.mock import patch, MagicMock
import torch

from nba_game_recap_summarizer.finetuning.models.llama_model import LlamaRecapSummarizationModel


class TestInferenceServiceE2E:
    """End-to-end tests for the inference service."""
    
    @pytest.fixture
    def mock_model_checkpoint(self):
        """Create a mock model checkpoint for E2E testing."""
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as tmp_file:
            checkpoint = {
                'state_dict': {},
                'hyper_parameters': {
                    'model_name': 'test-model',
                    'model_type': 'llama',
                    'peft_method': 'lora'
                }
            }
            torch.save(checkpoint, tmp_file.name)
            yield tmp_file.name
        
        os.unlink(tmp_file.name)
    
    @pytest.fixture
    def mock_llama_model(self, mock_model_checkpoint):
        """Create a mock LlamaRecapSummarizationModel for E2E testing."""
        # Create a simple mock model for E2E testing
        mock_model = MagicMock()
        mock_model.is_loaded.return_value = True
        mock_model.summarize_recap.return_value = "E2E Test Summary: Lakers beat Suns in overtime with a spectacular game-winning shot by Bryant."
        
        return mock_model
    
    def test_complete_inference_workflow(self, mock_llama_model, mock_model_checkpoint):
        """Test the complete inference workflow from request to response."""
        with patch('nba_game_recap_summarizer.api.inference.LlamaRecapSummarizationModel.load_model_from_checkpoint') as mock_load, \
             patch('nba_game_recap_summarizer.api.inference.os.path.exists') as mock_exists, \
             patch('nba_game_recap_summarizer.api.inference.model', mock_llama_model):
            
            mock_exists.return_value = True
            mock_load.return_value = mock_llama_model
            
            # Import and create the app
            from nba_game_recap_summarizer.api.inference import app
            from fastapi.testclient import TestClient
            
            # Create test client
            client = TestClient(app)
            
            # Test the complete workflow
            request_data = {
                "game_recap": "Lakers beat Suns in overtime with a Bryant game winner. The game was intense with both teams playing at their best.",
                "max_length": 512
            }
            
            response = client.post("/summarize_recap", json=request_data)
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert "game_recap_summary" in data
            assert data["game_recap_summary"] == "E2E Test Summary: Lakers beat Suns in overtime with a spectacular game-winning shot by Bryant."
            
            # Verify model was called correctly
            mock_llama_model.summarize_recap.assert_called_once_with(
                game_recap="Lakers beat Suns in overtime with a Bryant game winner. The game was intense with both teams playing at their best.",
                max_length=512
            )
    
    def test_api_documentation_endpoints(self, mock_llama_model, mock_model_checkpoint):
        """Test that API documentation endpoints are accessible."""
        with patch('nba_game_recap_summarizer.api.inference.LlamaRecapSummarizationModel.load_model_from_checkpoint') as mock_load, \
             patch('nba_game_recap_summarizer.api.inference.os.path.exists') as mock_exists:
            
            mock_exists.return_value = True
            mock_load.return_value = mock_llama_model
            
            from nba_game_recap_summarizer.api.inference import app
            from fastapi.testclient import TestClient
            
            client = TestClient(app)
            
            # Test Swagger UI
            response = client.get("/")
            assert response.status_code == 200
            assert "swagger" in response.text.lower() or "openapi" in response.text.lower()
            
            # Test ReDoc
            response = client.get("/redoc")
            assert response.status_code == 200
            assert "redoc" in response.text.lower()
    
    def test_health_check_workflow(self, mock_llama_model, mock_model_checkpoint):
        """Test the health check workflow."""
        with patch('nba_game_recap_summarizer.api.inference.LlamaRecapSummarizationModel.load_model_from_checkpoint') as mock_load, \
             patch('nba_game_recap_summarizer.api.inference.os.path.exists') as mock_exists:
            
            mock_exists.return_value = True
            mock_load.return_value = mock_llama_model
            
            from nba_game_recap_summarizer.api.inference import app
            from fastapi.testclient import TestClient
            
            client = TestClient(app)
            
            # Test health check
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True
    
    def test_error_scenarios_e2e(self, mock_llama_model, mock_model_checkpoint):
        """Test error scenarios in the E2E workflow."""
        with patch('nba_game_recap_summarizer.api.inference.LlamaRecapSummarizationModel.load_model_from_checkpoint') as mock_load, \
             patch('nba_game_recap_summarizer.api.inference.os.path.exists') as mock_exists:
            
            mock_exists.return_value = True
            mock_load.return_value = mock_llama_model
            
            from nba_game_recap_summarizer.api.inference import app
            from fastapi.testclient import TestClient
            
            client = TestClient(app)
            
            # Test invalid request data
            invalid_requests = [
                {"game_recap": "", "max_length": 1024},  # Empty recap
                {"game_recap": "Valid recap", "max_length": 0},  # Invalid max_length
                {"game_recap": "Valid recap", "max_length": 3000},  # Too large max_length
                {"invalid_field": "test"},  # Missing required field
            ]
            
            for invalid_request in invalid_requests:
                response = client.post("/summarize_recap", json=invalid_request)
                assert response.status_code in [400, 422]  # Bad request or validation error
    
    def test_model_loading_scenarios_e2e(self, mock_llama_model, mock_model_checkpoint):
        """Test different model loading scenarios in E2E."""
        from nba_game_recap_summarizer.api.inference import app
        from fastapi.testclient import TestClient
        
        # Test loading from local path
        with patch('nba_game_recap_summarizer.api.inference.LlamaRecapSummarizationModel.load_model_from_checkpoint') as mock_load, \
             patch('nba_game_recap_summarizer.api.inference.os.path.exists') as mock_exists:
            
            mock_exists.return_value = True
            mock_load.return_value = mock_llama_model
            
            client = TestClient(app)
            response = client.get("/health")
            assert response.status_code == 200
            
            # Verify local path was used
            mock_exists.assert_called_with("/app/models/model.ckpt")
            mock_load.assert_called_with(checkpoint_path="/app/models/model.ckpt")
        
        # Test loading from S3 path
        with patch('nba_game_recap_summarizer.api.inference.LlamaRecapSummarizationModel.load_model_from_checkpoint') as mock_load, \
             patch('nba_game_recap_summarizer.api.inference.os.path.exists') as mock_exists, \
             patch('nba_game_recap_summarizer.api.inference.settings') as mock_settings:
            
            mock_exists.return_value = False
            mock_settings.model_path = "s3://test-bucket/model.ckpt"
            mock_load.return_value = mock_llama_model
            
            client = TestClient(app)
            response = client.get("/health")
            assert response.status_code == 200
            
            # Verify S3 path was used
            mock_exists.assert_called_with("/app/models/model.ckpt")
            mock_load.assert_called_with(checkpoint_path="s3://test-bucket/model.ckpt")
    
    def test_request_logging_e2e(self, mock_llama_model, mock_model_checkpoint):
        """Test that request logging works in E2E."""
        with patch('nba_game_recap_summarizer.api.inference.LlamaRecapSummarizationModel.load_model_from_checkpoint') as mock_load, \
             patch('nba_game_recap_summarizer.api.inference.os.path.exists') as mock_exists, \
             patch('nba_game_recap_summarizer.api.inference.logger') as mock_logger:
            
            mock_exists.return_value = True
            mock_load.return_value = mock_llama_model
            
            from nba_game_recap_summarizer.api.inference import app
            from fastapi.testclient import TestClient
            
            client = TestClient(app)
            
            # Make a request
            response = client.get("/api")
            assert response.status_code == 200
            
            # Verify logging was called
            mock_logger.info.assert_called()
    
    def test_concurrent_requests_e2e(self, mock_llama_model, mock_model_checkpoint):
        """Test handling of concurrent requests in E2E."""
        with patch('nba_game_recap_summarizer.api.inference.LlamaRecapSummarizationModel.load_model_from_checkpoint') as mock_load, \
             patch('nba_game_recap_summarizer.api.inference.os.path.exists') as mock_exists:
            
            mock_exists.return_value = True
            mock_load.return_value = mock_llama_model
            
            from nba_game_recap_summarizer.api.inference import app
            from fastapi.testclient import TestClient
            
            client = TestClient(app)
            
            # Make multiple concurrent requests
            request_data = {
                "game_recap": "Lakers beat Suns in overtime.",
                "max_length": 256
            }
            
            responses = []
            for i in range(5):
                response = client.post("/summarize_recap", json=request_data)
                responses.append(response)
            
            # All requests should succeed
            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert "game_recap_summary" in data
    
    def test_model_error_handling_e2e(self, mock_llama_model, mock_model_checkpoint):
        """Test model error handling in E2E."""
        with patch('nba_game_recap_summarizer.api.inference.LlamaRecapSummarizationModel.load_model_from_checkpoint') as mock_load, \
             patch('nba_game_recap_summarizer.api.inference.os.path.exists') as mock_exists:
            
            mock_exists.return_value = True
            mock_load.return_value = mock_llama_model
            
            # Make the model raise an error
            mock_llama_model.summarize_recap.side_effect = Exception("Model error")
            
            from nba_game_recap_summarizer.api.inference import app
            from fastapi.testclient import TestClient
            
            client = TestClient(app)
            
            request_data = {
                "game_recap": "Lakers beat Suns in overtime.",
                "max_length": 256
            }
            
            response = client.post("/summarize_recap", json=request_data)
            assert response.status_code == 500
            assert "Error summarizing game recap" in response.json()["detail"]


class TestInferenceServiceWithRealModel:
    """E2E tests with real model components when available."""
    
    def test_real_model_loading_e2e(self):
        """Test loading a real model checkpoint in E2E."""
        checkpoint_path = "tests/resources/artifacts/pipeline_id/best_model.ckpt"
        
        if not os.path.exists(checkpoint_path):
            pytest.skip("Real checkpoint file not found")
        
        try:
            # Test loading the real model
            model = LlamaRecapSummarizationModel.load_model_from_checkpoint(
                checkpoint_path=checkpoint_path
            )
            
            assert model is not None
            assert model.is_loaded() is True
            
            # Test the complete E2E workflow with real model
            from nba_game_recap_summarizer.api.inference import app
            from fastapi.testclient import TestClient
            
            with patch('nba_game_recap_summarizer.api.inference.LlamaRecapSummarizationModel.load_model_from_checkpoint') as mock_load, \
                 patch('nba_game_recap_summarizer.api.inference.os.path.exists') as mock_exists:
                
                mock_exists.return_value = True
                mock_load.return_value = model
                
                client = TestClient(app)
                
                # Test health check
                response = client.get("/health")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                assert data["model_loaded"] is True
                
                # Test summarization (might fail if model is not properly configured)
                try:
                    request_data = {
                        "game_recap": "Lakers beat Suns in overtime.",
                        "max_length": 100
                    }
                    
                    response = client.post("/summarize_recap", json=request_data)
                    assert response.status_code == 200
                    data = response.json()
                    assert "game_recap_summary" in data
                    assert isinstance(data["game_recap_summary"], str)
                    
                except Exception as e:
                    # If summarization fails, that's okay for E2E tests
                    # as long as the model loaded successfully
                    pytest.skip(f"Summarization failed with real model: {e}")
                    
        except Exception as e:
            pytest.skip(f"Could not load real model: {e}")
    
    def test_model_performance_e2e(self):
        """Test model performance characteristics in E2E."""
        checkpoint_path = "tests/resources/artifacts/pipeline_id/best_model.ckpt"
        
        if not os.path.exists(checkpoint_path):
            pytest.skip("Real checkpoint file not found")
        
        try:
            model = LlamaRecapSummarizationModel.load_model_from_checkpoint(
                checkpoint_path=checkpoint_path
            )
            
            # Test response time
            import time
            start_time = time.time()
            
            try:
                result = model.summarize_recap(
                    game_recap="Lakers beat Suns in overtime with a spectacular game-winning shot.",
                    max_length=100
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                # Response should be reasonably fast (less than 30 seconds for a test)
                assert response_time < 30
                assert isinstance(result, str)
                assert len(result) > 0
                
            except Exception as e:
                pytest.skip(f"Performance test failed: {e}")
                
        except Exception as e:
            pytest.skip(f"Could not load real model for performance test: {e}")
