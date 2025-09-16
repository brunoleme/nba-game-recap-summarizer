# Phi-3.5-mini Testing Guide

This document explains how to test both LLaMA and Phi-3.5-mini models in the NBA Game Recap Summarizer project.

## Test Configurations

### 1. Unit Tests (Fast)
- **File**: `tests/unit/test_models.py`
- **Models**: Both LLaMA and Phi models
- **Speed**: ⚡⚡⚡ Very Fast
- **Use Case**: Basic functionality validation

```bash
pytest tests/unit/test_models.py -v
```

### 2. Integration Tests (Medium)
- **File**: `tests/integration/test_phi_pipeline.py`
- **Models**: Phi model with TinyLlama substitute
- **Speed**: ⚡⚡ Fast
- **Use Case**: Architecture and pipeline testing

```bash
pytest tests/integration/test_phi_pipeline.py -v
```

### 3. Comprehensive Integration Tests (Medium)
- **File**: `tests/test_phi_integration.py`
- **Models**: Both LLaMA and Phi models
- **Speed**: ⚡⚡ Fast
- **Use Case**: Complete pipeline validation

```bash
pytest tests/test_phi_integration.py -v
```

### 4. End-to-End Tests (Slow)
- **File**: `tests/e2e/test_inference_e2e.py`
- **Models**: Both LLaMA and Phi models
- **Speed**: ⚡ Slow
- **Use Case**: Full API testing

```bash
pytest tests/e2e/test_inference_e2e.py -v
```

## Test Data Configurations

### 1. Tiny LLaMA (Fastest)
- **Config**: `tests/resources/config/config.test.yaml`
- **Model**: `hf-internal-testing/tiny-random-LlamaForCausalLM`
- **Use Case**: Unit tests, basic validation

### 2. TinyLlama as Phi Substitute (Fast)
- **Config**: `tests/resources/config/config.test.phi.tiny.yaml`
- **Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Use Case**: Phi architecture testing

### 3. Full Phi-3.5-mini (Slow)
- **Config**: `tests/resources/config/config.test.phi.yaml`
- **Model**: `microsoft/Phi-3.5-mini-instruct`
- **Use Case**: Full functionality testing

## Running All Tests

### Quick Test Suite (Recommended for Development)
```bash
# Run all unit tests
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Run comprehensive integration tests
pytest tests/test_phi_integration.py -v
```

### Full Test Suite (For CI/CD)
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/nba_game_recap_summarizer --cov-report=html
```

## Model-Specific Testing

### Testing LLaMA Models
```bash
# Test LLaMA-specific functionality
pytest tests/unit/test_models.py::test_llama_model_initialization -v
pytest tests/unit/test_models.py::test_llama_model_with_peft -v
```

### Testing Phi Models
```bash
# Test Phi-specific functionality
pytest tests/unit/test_models.py::test_phi_model_initialization -v
pytest tests/unit/test_models.py::test_phi_model_with_peft -v
```

### Testing Both Models Together
```bash
# Test model compatibility
pytest tests/test_phi_integration.py::test_model_compatibility -v
pytest tests/test_phi_integration.py::test_model_architecture_differences -v
```

## Configuration Testing

### Test Different Model Types
```bash
# Test with LLaMA config
pytest tests/ -k "llama" -v

# Test with Phi config
pytest tests/ -k "phi" -v
```

### Test Different Quantization Settings
```bash
# Test quantization
pytest tests/ -k "quantization" -v
```

### Test Different PEFT Methods
```bash
# Test LoRA
pytest tests/ -k "lora" -v
```

## Performance Testing

### Memory Usage Testing
```bash
# Test memory usage with different models
pytest tests/test_phi_integration.py::test_model_quantization_support -v
```

### Generation Speed Testing
```bash
# Test generation speed
pytest tests/integration/test_phi_pipeline.py::test_phi_model_generation -v
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Use smaller models or disable quantization
2. **Model Loading Errors**: Check model paths and dependencies
3. **Test Failures**: Ensure all dependencies are installed

### Debug Mode
```bash
# Run tests with debug output
pytest tests/ -v -s --tb=short
```

### Specific Test Debugging
```bash
# Debug specific test
pytest tests/unit/test_models.py::test_phi_model_initialization -v -s --tb=long
```

## Test Data

### Sample Data
- **File**: `tests/resources/source_data/game_recaps_with_summaries_sample.csv`
- **Size**: 3 samples
- **Use Case**: Basic testing

### Mock Data
- **Generated**: Automatically in tests
- **Use Case**: Unit testing

## Continuous Integration

### GitHub Actions
The project includes GitHub Actions workflows that run:
- Unit tests
- Integration tests
- Linting
- Type checking

### Local CI Simulation
```bash
# Run all checks locally
make test
make lint
make type-check
```

## Best Practices

1. **Use Tiny Models**: For fast development testing
2. **Mock Heavy Operations**: For unit tests
3. **Test Both Models**: Ensure compatibility
4. **Use Appropriate Configs**: Match test speed to use case
5. **Clean Up**: Remove temporary files after tests

## Model Comparison

| Model | Speed | Memory | Context | Use Case |
|-------|-------|--------|---------|----------|
| `hf-internal-testing/tiny-random-LlamaForCausalLM` | ⚡⚡⚡ | Low | 2048 | Unit tests |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ⚡⚡ | Medium | 2048 | Integration tests |
| `microsoft/Phi-3.5-mini-instruct` | ⚡ | High | 4096 | Full testing |

## Next Steps

1. **Run Tests**: Start with unit tests
2. **Check Coverage**: Ensure all code paths are tested
3. **Add New Tests**: As you add new features
4. **Update Configs**: As you modify model configurations
5. **Monitor Performance**: Track test execution times
