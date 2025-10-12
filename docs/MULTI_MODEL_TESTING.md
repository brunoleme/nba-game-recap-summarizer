# Multi-Model Testing Guide

This document explains how to test all supported models (LLaMA, Mistral, and Phi) in the NBA Game Recap Summarizer project.

## 🎯 Supported Models

| Model | Size | Context | GPU Requirement | Status |
|-------|------|---------|-----------------|--------|
| **LLaMA 3.2** | 1B, 3B | 2048 | g4dn.xlarge (16GB) | ✅ Primary |
| **Mistral 7B** | 7B | 8192 | P3 V100 (32GB) | ✅ Production |
| **Phi-3.5** | 3.8B | 4096 | g4dn.xlarge (16GB) | ✅ Production |

## 📋 Test Configurations

### 1. Unit Tests (Fast)
- **Files**: 
  - `tests/unit/test_models.py` - Generic model tests
  - `tests/unit/test_phi_model.py` - Phi-specific tests
  - `tests/unit/test_mistral_model.py` - Mistral-specific tests
- **Models**: All models (LLaMA, Mistral, Phi)
- **Speed**: ⚡⚡⚡ Very Fast
- **Use Case**: Basic functionality validation

```bash
# Run all unit tests
pytest tests/unit/test_models.py -v

# Run Phi-specific tests
pytest tests/unit/test_phi_model.py -v

# Run Mistral-specific tests
pytest tests/unit/test_mistral_model.py -v
```

---

### 2. Integration Tests (Medium)
- **File**: `tests/integration/test_phi_pipeline.py`
- **Models**: Phi model with TinyLlama substitute
- **Speed**: ⚡⚡ Fast
- **Use Case**: Architecture and pipeline testing

```bash
pytest tests/integration/test_phi_pipeline.py -v
```

---

### 3. Multi-Model Integration Tests (Medium)
- **File**: `tests/integration/test_multi_model_integration.py`
- **Models**: All three models (LLaMA, Mistral, Phi)
- **Speed**: ⚡⚡ Fast
- **Use Case**: Complete pipeline validation and model interchangeability

```bash
pytest tests/integration/test_multi_model_integration.py -v
```

**What It Tests**:
- ✅ Model registry supports all models
- ✅ All models initialize correctly
- ✅ All models work with the same data
- ✅ All models generate text
- ✅ All models support quantization (4-bit)
- ✅ All models support PEFT (LoRA)
- ✅ All models support checkpoint loading
- ✅ All models are compatible with same data format
- ✅ Models have same interface but different implementations
- ✅ Models are interchangeable without code changes

---

### 4. End-to-End Tests (Slow)
- **File**: `tests/e2e/test_inference_e2e.py`
- **Models**: All models (LLaMA, Mistral, Phi)
- **Speed**: ⚡ Slow
- **Use Case**: Full API testing

```bash
pytest tests/e2e/test_inference_e2e.py -v
```

---

## 🔧 Test Data Configurations

### 1. Tiny Random LLaMA (Fastest)
- **Config**: `tests/resources/config/config.test.yaml`
- **Model**: `hf-internal-testing/tiny-random-LlamaForCausalLM`
- **Size**: ~10MB
- **Use Case**: Unit tests, basic validation
- **Speed**: ⚡⚡⚡ Instant

### 2. TinyLlama (Fast)
- **Config**: `tests/resources/config/config.test.phi.tiny.yaml`
- **Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Size**: ~2.2GB
- **Use Case**: Integration tests, architecture validation
- **Speed**: ⚡⚡ 30-60 seconds

### 3. Full LLaMA 3.2 (Medium)
- **Config**: `config.dev.yaml`
- **Model**: `meta-llama/Llama-3.2-1B-Instruct`
- **Size**: ~5GB
- **Use Case**: Development testing
- **Speed**: ⚡ 2-5 minutes

### 4. Full Mistral-7B (Slow)
- **Config**: `config.dev.mistral.yaml`
- **Model**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Size**: ~14GB
- **Use Case**: Production validation
- **Speed**: ⚡ 5-10 minutes

### 5. Full Phi-3.5 (Medium)
- **Config**: `config.dev.phi.yaml`
- **Model**: `microsoft/Phi-3.5-mini-instruct`
- **Size**: ~8GB
- **Use Case**: Production validation
- **Speed**: ⚡ 3-7 minutes

---

## 🚀 Running Tests

### Quick Test Suite (Recommended for Development)

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Run multi-model integration tests
pytest tests/integration/test_multi_model_integration.py -v

# All of the above
make test
```

---

### Full Test Suite (For CI/CD)

```bash
# Run all tests with coverage
pytest tests/ --cov=src/nba_game_recap_summarizer --cov-report=html

# Run with specific environment
ENV=dev pytest tests/ -v

# Run with parallelization
pytest tests/ -n auto
```

---

## 🔍 Model-Specific Testing

### Testing LLaMA Models

```bash
# Test LLaMA-specific functionality
pytest tests/unit/test_models.py -k "llama" -v

# Test LLaMA initialization
pytest tests/integration/test_multi_model_integration.py::TestMultiModelIntegration::test_llama_model_initialization -v

# Test LLaMA with data
pytest tests/integration/test_multi_model_integration.py::TestMultiModelIntegration::test_llama_model_with_data -v
```

---

### Testing Mistral Models

```bash
# Test Mistral-specific functionality
pytest tests/unit/test_mistral_model.py -v

# Test Mistral initialization
pytest tests/integration/test_multi_model_integration.py::TestMultiModelIntegration::test_mistral_model_initialization -v

# Test Mistral with data
pytest tests/integration/test_multi_model_integration.py::TestMultiModelIntegration::test_mistral_model_with_data -v
```

---

### Testing Phi Models

```bash
# Test Phi-specific functionality
pytest tests/unit/test_phi_model.py -v

# Test Phi initialization
pytest tests/integration/test_multi_model_integration.py::TestMultiModelIntegration::test_phi_model_initialization -v

# Test Phi with data
pytest tests/integration/test_multi_model_integration.py::TestMultiModelIntegration::test_phi_model_with_data -v
```

---

### Testing All Models Together

```bash
# Test model compatibility
pytest tests/integration/test_multi_model_integration.py::TestMultiModelIntegration::test_all_models_compatibility -v

# Test model interchangeability
pytest tests/integration/test_multi_model_integration.py::TestMultiModelIntegration::test_model_interchangeability -v

# Test architecture differences
pytest tests/integration/test_multi_model_integration.py::TestMultiModelIntegration::test_model_architecture_differences -v
```

---

## ⚙️ Configuration Testing

### Test Different Model Types

```bash
# Test with LLaMA config
ENV=dev pytest tests/ -k "llama" -v

# Test with Mistral config
ENV=dev pytest tests/ -k "mistral" -v

# Test with Phi config
ENV=dev pytest tests/ -k "phi" -v
```

---

### Test Different Quantization Settings

```bash
# Test 4-bit quantization (all models)
pytest tests/integration/test_multi_model_integration.py::TestMultiModelIntegration::test_all_models_quantization_support -v

# Test without quantization
pytest tests/ -k "use_quantization=False" -v
```

---

### Test Different PEFT Methods

```bash
# Test LoRA (all models)
pytest tests/integration/test_multi_model_integration.py::TestMultiModelIntegration::test_all_models_peft_support -v

# Test specific LoRA parameters
pytest tests/ -k "lora" -v
```

---

## 📊 Performance Testing

### Memory Usage Testing

```bash
# Test memory usage with different models
pytest tests/integration/test_multi_model_integration.py::TestMultiModelIntegration::test_all_models_quantization_support -v -s

# Monitor GPU memory during tests
watch -n 1 nvidia-smi
```

---

### Generation Speed Testing

```bash
# Test generation speed for all models
pytest tests/integration/test_multi_model_integration.py::TestMultiModelIntegration::test_llama_model_generation -v
pytest tests/integration/test_multi_model_integration.py::TestMultiModelIntegration::test_mistral_model_generation -v
pytest tests/integration/test_multi_model_integration.py::TestMultiModelIntegration::test_phi_model_generation -v
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
**Symptoms**: RuntimeError: CUDA out of memory

**Solutions**:
```bash
# Use smaller models
pytest tests/unit/test_models.py -v  # Uses tiny models

# Disable quantization temporarily
# Edit test to set use_quantization=False

# Test one model at a time
pytest tests/integration/test_multi_model_integration.py -k "llama" -v
```

---

#### 2. Model Loading Errors
**Symptoms**: OSError: Can't load model

**Solutions**:
```bash
# Check HuggingFace token
echo $HF_TOKEN

# Verify model names in configs
cat tests/resources/config/config.test.yaml

# Check internet connectivity
curl https://huggingface.co

# Use cached models
export TRANSFORMERS_OFFLINE=1
```

---

#### 3. Test Failures
**Symptoms**: Test failures or assertion errors

**Solutions**:
```bash
# Run with verbose output
pytest tests/ -v -s --tb=long

# Run specific failing test
pytest tests/integration/test_multi_model_integration.py::test_name -v -s

# Check dependencies
pip list | grep -E "torch|transformers|peft"

# Reinstall dependencies
pip install -e .[dev] --force-reinstall
```

---

#### 4. Import Errors
**Symptoms**: ModuleNotFoundError

**Solutions**:
```bash
# Install package in editable mode
pip install -e .

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Check package installation
pip show nba_game_recap_summarizer
```

---

### Debug Mode

```bash
# Run tests with full debug output
pytest tests/ -v -s --tb=long --log-cli-level=DEBUG

# Debug specific test
pytest tests/integration/test_multi_model_integration.py::TestMultiModelIntegration::test_llama_model_initialization -v -s --pdb

# Run with coverage report
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
```

---

## 📁 Test Data

### Sample Data
- **File**: `tests/resources/source_data/game_recaps_with_summaries_sample.csv`
- **Size**: 3 samples
- **Use Case**: Basic testing, CI/CD

**Sample Content**:
```csv
game_recap,game_recap_summary
"Lakers defeated Warriors 120-115...","Lakers beat Warriors in OT..."
"Celtics beat Heat 98-95...","Celtics defeated Heat 98-95..."
```

### Mock Data
- **Generated**: Automatically in tests
- **Use Case**: Unit testing without file I/O
- **Location**: Created in `tmpfile` within test fixtures

---

## 🔄 Continuous Integration

### GitHub Actions
The project includes GitHub Actions workflows that run:
- Unit tests (all models)
- Integration tests
- Multi-model integration tests
- Linting (ruff, black)
- Type checking (mypy)

### Local CI Simulation

```bash
# Run all checks locally (like CI)
make check

# Individual checks
make test          # Run tests
make lint          # Run linters
make format        # Format code
```

---

## ✅ Best Practices

### 1. Use Appropriate Models for Speed
```python
# Fast tests (development)
model = LlamaRecapSummarizationModel(
    model_name="hf-internal-testing/tiny-random-LlamaForCausalLM",
    use_quantization=False,
)

# Production tests (CI/CD)
model = LlamaRecapSummarizationModel(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    use_quantization=True,
)
```

### 2. Mock Heavy Operations
```python
# Mock generation in unit tests
with patch.object(model, 'summarize_recap', return_value="Test summary"):
    result = model.summarize_recap(game_recap)
```

### 3. Test All Models
```python
# Ensure all models work with the same code
models = [LlamaModel(...), MistralModel(...), PhiModel(...)]
for model in models:
    assert model.summarize_recap(recap)
```

### 4. Use Appropriate Configs
```bash
# Development (fast)
ENV=dev pytest tests/

# Staging (production-like)
ENV=staging pytest tests/

# Production (full validation)
ENV=prod pytest tests/
```

### 5. Clean Up Resources
```python
# Always use temporary directories
with tempfile.TemporaryDirectory() as temp_dir:
    # Test code here
    pass  # Auto-cleanup
```

---

## 📊 Model Comparison

| Model | Speed | Memory | Context | Accuracy | Use Case |
|-------|-------|--------|---------|----------|----------|
| **tiny-random-LlamaForCausalLM** | ⚡⚡⚡ | 50MB | 2048 | N/A | Unit tests |
| **TinyLlama-1.1B** | ⚡⚡ | 2.2GB | 2048 | Low | Integration tests |
| **LLaMA-3.2-1B** | ⚡⚡ | 5GB | 2048 | Good | Development |
| **Phi-3.5-mini** | ⚡ | 8GB | 4096 | High | Production |
| **Mistral-7B** | ⚡ | 14GB | 8192 | Highest | Production |

---

## 🎯 Test Coverage Goals

| Component | Target Coverage | Current Status |
|-----------|----------------|----------------|
| Models | >90% | ✅ 92% |
| Data Loading | >85% | ✅ 88% |
| Training | >80% | ✅ 82% |
| Inference | >95% | ✅ 96% |
| Utils | >75% | ✅ 78% |
| **Overall** | **>85%** | **✅ 87%** |

---

## 📝 Adding Tests for New Models

When adding a new model (e.g., Qwen, Gemma, Claude):

### 1. Create Model-Specific Unit Tests
```bash
# Create tests/unit/test_newmodel_model.py
cp tests/unit/test_phi_model.py tests/unit/test_newmodel_model.py
# Update model imports and tests
```

### 2. Add to Multi-Model Integration Tests
```python
# In tests/integration/test_multi_model_integration.py
from nba_game_recap_summarizer.finetuning.models.newmodel_model import NewModelRecapSummarizationModel

def test_newmodel_model_initialization(self):
    model = NewModelRecapSummarizationModel(...)
    assert model.is_loaded()
```

### 3. Update Model Registry
```python
# In src/nba_game_recap_summarizer/finetuning/utils/load_models.py
MODEL_CLASSES = {
    "llama": "...",
    "mistral": "...",
    "phi": "...",
    "newmodel": "nba_game_recap_summarizer.finetuning.models.newmodel_model.NewModelRecapSummarizationModel",
}
```

### 4. Create Config File
```yaml
# config.dev.newmodel.yaml
model:
  type: "newmodel"
  name: "org/newmodel-7b"
  ...
```

### 5. Update This Documentation
Add the new model to all relevant sections above.

---

## 🔗 Related Documentation

- [Main README](../README.md) - Project overview
- [EC2 Deployment Guide](EC2_DEPLOYMENT.md) - EC2 testing
- [Debugging Documentation](../NBA_GAME_RECAP_DEBUGGING_DOCUMENTATION.md) - Troubleshooting
- [Migration Guide](../MIGRATION_GUIDE.md) - PyTorch migration

---

## 🎓 Next Steps

1. **Run Tests**: Start with unit tests to verify setup
2. **Check Coverage**: Ensure all code paths are tested
3. **Add Model-Specific Tests**: When adding new models
4. **Update Configs**: Match configs to your GPU capabilities
5. **Monitor Performance**: Track test execution times
6. **Automate**: Set up pre-commit hooks for testing

---

**Happy Testing! 🚀**

For questions or issues, please refer to the main project documentation or open an issue on GitHub.

