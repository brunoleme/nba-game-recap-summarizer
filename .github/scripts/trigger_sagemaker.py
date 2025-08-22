import argparse
import boto3
import datetime
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-uri", required=True, help="ECR image URI")
    parser.add_argument("--role-arn", required=True, help="SageMaker Execution Role ARN")
    parser.add_argument("--job-name", required=True, help="Training Job Name")
    parser.add_argument("--instance-type", default="ml.m5.large", help="Instance type")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--env", required=True, choices=["dev", "staging", "prod"], help="Environment")
    parser.add_argument("--wandb-api-key", required=True, help="W&B API Key")
    return parser.parse_args()


def main():
    args = parse_args()

    client = boto3.client("sagemaker")

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    training_job_name = f"{args.job_name}-{timestamp}"

    response = client.create_training_job(
        TrainingJobName=training_job_name,
        AlgorithmSpecification={
            "TrainingImage": args.image_uri,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["python", "scripts/train.py", "--config-path=src/text2cypher/finetuning/config", f"--config-name=config.{args.env}"],
        },
        RoleArn=args.role_arn,
        InputDataConfig=[
            {
                'ChannelName': 'training',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': f"s3://bl-portfolio-ml-sagemaker-source-data/notechat-dataset/notechat_dataset.csv",
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                },
                'ContentType': 'application/json',
                'InputMode': 'File'
            }
        ],
        OutputDataConfig={
            "S3OutputPath": f"s3://bl-portfolio-ml-sagemaker-{args.env}/output/{training_job_name}/"
        },
        ResourceConfig={
            "InstanceType": args.instance_type,
            "InstanceCount": args.instance_count,
            "VolumeSizeInGB": 30,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
        Environment={
            "ENV": args.env,
            "WANDB_API_KEY": args.wandb_api_key
        },
    )

    print(f"Triggered training job: {training_job_name}")

    # Wait until the training job completes
    client.get_waiter("training_job_completed_or_stopped").wait(TrainingJobName=training_job_name)

    desc = client.describe_training_job(TrainingJobName=training_job_name)
    model_artifact_uri = desc["ModelArtifacts"]["S3ModelArtifacts"]

    # Output for GitHub Actions
    with open(os.environ["GITHUB_ENV"], "a") as f:
        f.write(f"MODEL_ARTIFACT_S3_URI={model_artifact_uri}\n")

    # Write to GITHUB_OUTPUT for outputs (required for 'outputs' keyword)
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"model_artifact_s3_uri={model_artifact_uri}\n")
    print(model_artifact_uri)


if __name__ == "__main__":
    main()
