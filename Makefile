.PHONY: install test lint format clean clean-artifacts check run train evaluate clean-venv create-venv clean-all sagemaker-trigger sagemaker-pipeline-trigger

ENV ?= dev
CONFIG_PATH ?= src/text2cypher/finetuning/config
CONFIG_FILE := $(CONFIG_PATH)/config.$(ENV).yaml

help:
	@echo "Makefile commands:"
	@echo "  install                      Install dependencies with uv"
	@echo "  test                         Run tests"
	@echo "  lint                         Run code linters"
	@echo "  format                       Auto-format code"
	@echo "  clean                        Remove Python and test cache"
	@echo "  clean-artifacts              Remove checkpoint artifacts"
	@echo "  run                           Run the full pipeline (default: ENV=dev)"
	@echo "  train                         Run only training"
	@echo "  evaluate                      Run only evaluation"
	@echo "  docker-build                  Build Docker image for ECR"
	@echo "  docker-push                   Push Docker image to ECR"
	@echo "  sagemaker-trigger             Trigger SageMaker training job"
	@echo "  sagemaker-pipeline-trigger    Trigger SageMaker preprocess-train-eval-deploy pipeline"
	@echo "  sagemaker-deploy-endpoint     Deploy model to SageMaker Endpoint"


ARTIFACTS_DIR := $(shell \
	if [ -f config.yaml ]; then \
		python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['training']['checkpoint_dir'])"; \
	else \
		echo "checkpoints"; \
	fi)

install:
	pip install -e .[dev]

test:
	PYTHONPATH=. ENV=$(ENV) pytest tests/ --cov=src --cov-report=term-missing -s

lint:
	ruff check .
	black . --check
	mypy src/

format:
	black .
	ruff check . --fix

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name "*.so" -delete
	find . -type f -name ".coverage*" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +


clean-artifacts:
	rm -rf $(ARTIFACTS_DIR)/*

clean-venv:
	rm -rf .venv

create-venv:
	python3.9 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip setuptools

clean-all: clean clean-venv

check: lint test

check-env:
	@test -f $(CONFIG_FILE) || (echo "Missing config file: $(CONFIG_FILE)" && exit 1)

train:
	PYTHONPATH=. ENV=$(ENV) python scripts/train.py

evaluate:
	PYTHONPATH=. ENV=$(ENV) python scripts/evaluate_model.py

run: check-env
	PYTHONPATH=. ENV=$(ENV) python scripts/run_pipeline.py

docker-build:
	docker build -t $(ECR_REPOSITORY_URI):$(IMAGE_TAG) .

docker-push:
	docker push $(ECR_REPOSITORY_URI):$(IMAGE_TAG)

sagemaker-trigger:
	@MODEL_URI=$$(python .github/scripts/trigger_sagemaker.py \
		--image-uri $(ECR_REPOSITORY_URI):$(IMAGE_TAG) \
		--role-arn $(SAGEMAKER_ROLE_ARN) \
		--job-name text2cypher-$(ENV) \
		--env $(ENV) \
		--wandb-api-key $(WANDB_API_KEY) \
		--instance-type ml.g4dn.xlarge)

PREPROCESSING_INSTANCE_TYPE ?= ml.g4dn.xlarge
TRAINING_INSTANCE_TYPE ?= ml.g4dn.xlarge
EVALUATION_INSTANCE_TYPE ?= ml.g4dn.xlarge
DEPLOYMENT_INSTANCE_TYPE ?= ml.g4dn.xlarge

sagemaker-pipeline-trigger:
	@echo "Running SageMaker trigger..."
	python .github/scripts/trigger_sagemaker_pipeline.py \
		--pipeline-run-id $(PIPELINE_RUN_ID) \
		--image-uri $(ECR_REPOSITORY_URI):$(IMAGE_TAG) \
		--inference-image-uri $(ECR_REPOSITORY_URI):$(INFERENCE_IMAGE_TAG) \
		--role-arn $(SAGEMAKER_ROLE_ARN) \
		--job-name text2cypher-$(ENV) \
		--env $(ENV) \
		--wandb-api-key $(WANDB_API_KEY) \
		--openai-api-key $(OPENAI_API_KEY) \
		--preprocessing-instance-type $(PREPROCESSING_INSTANCE_TYPE) \
		--preprocessing-instance-count 1 \
		--training-instance-type $(TRAINING_INSTANCE_TYPE) \
		--training-instance-count 1 \
		--evaluation-instance-type $(EVALUATION_INSTANCE_TYPE) \
		--evaluation-instance-count 1 \
		--deployment-instance-type $(DEPLOYMENT_INSTANCE_TYPE)
