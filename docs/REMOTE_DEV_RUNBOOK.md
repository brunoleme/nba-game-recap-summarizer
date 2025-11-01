### Remote Dev Runbook (AWS host)

Purpose: run, test, and iterate quickly on a remote AWS machine (GPU EC2) for this repo.

#### 1) Prereqs on the remote host
- OS: Ubuntu 20.04/22.04 recommended
- GPU: NVIDIA driver + CUDA working (verify with `nvidia-smi`)
- Tools: `git`, `python3.10` + `python3.10-venv` (or 3.9+), `tmux`, `docker` (optional for containers), `awscli` v2
- Disk: ≥ 50 GB free (models/artifacts)

Quick install (Ubuntu):
```bash
sudo apt update -y && sudo apt install -y git tmux python3.10 python3.10-venv build-essential
# Optional: Docker (if you plan to build images on the host)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

#### 2) Clone and branch
```bash
git clone https://github.com/brunoleme/nba-game-recap-summarizer.git
cd nba-game-recap-summarizer
git checkout dev
```

#### 3) Python environment
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
# If you prefer uv:
# pip install uv && uv pip install -e .[dev]
```

Verify critical libs (align with notebooks/pipeline):
```bash
python - << 'PY'
import transformers, trl, torch
print('transformers', transformers.__version__)
print('trl', trl.__version__)
print('torch', torch.__version__)
PY
```

#### 4) Auth and environment variables (never commit secrets)
```bash
export ENV=dev
export AWS_REGION=<region>
export AWS_ACCOUNT_ID=<acct>
export WANDB_API_KEY=<wandb>
export OPENAI_API_KEY=<openai>
export HF_TOKEN=<hf>
export HUGGINGFACEHUB_API_TOKEN=$HF_TOKEN

# Optional for ECR/Docker builds
export ECR_REPOSITORY_URI=<acct>.dkr.ecr.${AWS_REGION}.amazonaws.com/nba-recap
export IMAGE_TAG=dpo-latest
export SAGEMAKER_ROLE_ARN=arn:aws:iam::<acct>:role/<role>

# Base supervised model artifacts for DPO (dev/staging/prod bucket)
export BASE_MODEL_PATH=s3://nba-recap-summarization-model-dev/output/artifacts/<pipeline_id>/hf_model_merged
```

AWS credentials: use `aws configure` or an assumed role mechanism (aws-vault/SSO). Ensure `aws sts get-caller-identity` works.

#### 5) Fast iteration with tmux
```bash
tmux new -s recap
# split panes: Ctrl+b %  (vertical), Ctrl+b " (horizontal)
# detach: Ctrl+b d
tmux ls
tmux attach -t recap
```

#### 6) Local smoke tests
```bash
# Unit/integration slices
pytest -q tests/unit/test_inference.py -q
pytest -q tests/integration/test_preprocessing_pipeline.py -q
```

#### 7) Run DPO locally (GPU)
- Configs live under `src/nba_game_recap_summarizer/finetuning/config/` (e.g., `config.dpo.dev.yaml`).

Preprocess pairs (CSV path is configured in YAML):
```bash
PYTHONPATH=. ENV=dev python scripts/dpo_preprocessing.py
```

Run DPO tuning:
```bash
PYTHONPATH=. ENV=dev python scripts/dpo_tune.py
```

Evaluate tuned model:
```bash
PYTHONPATH=. ENV=dev python scripts/evaluate_dpo.py
```

Outputs are written under `/opt/ml/processing/output/model-artifacts` (configurable in YAML). For local runs without SageMaker, these are regular directories on disk; ensure the paths exist or adjust in the YAMLs.

#### 8) Inference API (with Prometheus metrics)
```bash
PYTHONPATH=. uvicorn src.nba_game_recap_summarizer.api.inference:app \
  --host 0.0.0.0 --port 8000 --log-level info

# Health and metrics
curl -s http://localhost:8000/health | jq
curl -s http://localhost:8000/metrics | head

# Summarize sample
curl -s -X POST http://localhost:8000/summarize_recap \
  -H 'Content-Type: application/json' \
  -d '{"game_recap":"Lakers beat Suns in OT on a Bryant game winner.","max_length":512}' | jq
```

#### 9) SageMaker DPO pipeline from host (optional)
```bash
make sagemaker-pipeline-trigger-dpo \
  PIPELINE_RUN_ID=$(uuidgen | tr 'A-Z' 'a-z') \
  ECR_REPOSITORY_URI=$ECR_REPOSITORY_URI \
  IMAGE_TAG=$IMAGE_TAG \
  SAGEMAKER_ROLE_ARN=$SAGEMAKER_ROLE_ARN \
  ENV=dev \
  WANDB_API_KEY=$WANDB_API_KEY \
  OPENAI_API_KEY=$OPENAI_API_KEY \
  HF_TOKEN=$HF_TOKEN \
  HUGGINGFACEHUB_API_TOKEN=$HUGGINGFACEHUB_API_TOKEN \
  PREPROCESSING_INSTANCE_TYPE=ml.m5.large \
  TRAINING_INSTANCE_TYPE=ml.g5.xlarge \
  EVALUATION_INSTANCE_TYPE=ml.g5.xlarge \
  DEPLOYMENT_INSTANCE_TYPE=ml.g4dn.xlarge \
  PROJECT_CONFIG=config.dpo.dev \
  BASE_MODEL_PATH=$BASE_MODEL_PATH
```

#### 10) Docker build/push (if needed)
```bash
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $ECR_REPOSITORY_URI

docker build -t $ECR_REPOSITORY_URI:$IMAGE_TAG -f Dockerfile.training .
docker push $ECR_REPOSITORY_URI:$IMAGE_TAG
```

#### 11) Observability quick checks
- API metrics at `/metrics` (Prometheus text format)
- W&B runs appear under the configured project
- CloudWatch: verify log groups for pipelines/jobs if running in AWS

#### 12) Common issues
- CUDA OOM: reduce batch size/accumulation in YAML, ensure `nvidia-smi` sees free memory
- BitsAndBytes errors: ensure GPU present; fall back to non-4bit by setting `model.quantization: false`
- TRL/Transformers mismatches: ensure `trl>=0.24.0` and compatible `transformers`
- Tokenizer pad token: code sets `pad_token=eos_token` if missing
- TorchVision warnings: safe to ignore if not using image I/O

#### 13) Safety & secrets
- Never commit secrets; pass via env vars or AWS Secrets Manager/SSM
- Keep `/metrics` non-public; restrict via security groups/VPC

#### 14) Handy paths
- Configs: `src/nba_game_recap_summarizer/finetuning/config/`
- DPO code: `src/nba_game_recap_summarizer/finetuning/`
- Scripts: `scripts/`
- API: `src/nba_game_recap_summarizer/api/inference.py`

This runbook is intentionally concise so iteration can be fast on a remote GPU host.


