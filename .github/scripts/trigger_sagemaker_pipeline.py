import argparse
import uuid
from sagemaker_pipeline import create_pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-uri", required=True, help="ECR image URI")
    parser.add_argument("--inference-image-uri", required=True, help="ECR inference image URI")
    parser.add_argument("--role-arn", required=True, help="SageMaker Execution Role ARN")
    parser.add_argument("--job-name", required=True, help="Training Job Name")
    parser.add_argument("--pipeline-run-id", required=True, help="Pipeline UUID")
    parser.add_argument("--env", required=True, choices=["dev", "staging", "prod"], help="Environment")
    parser.add_argument("--wandb-api-key", required=True, help="W&B API Key")
    parser.add_argument("--openai-api-key", required=True, help="Open AI API Key")
    parser.add_argument("--preprocessing-instance-type", default="ml.m5.large", help="Instance type")
    parser.add_argument("--preprocessing-instance-count", type=int, default=1)
    parser.add_argument("--training-instance-type", default="ml.m5.large", help="Instance type")
    parser.add_argument("--training-instance-count", type=int, default=1)
    parser.add_argument("--evaluation-instance-type", default="ml.m5.large", help="Instance type")
    parser.add_argument("--evaluation-instance-count", type=int, default=1)
    parser.add_argument("--deployment-instance-type", default="ml.m5.large", help="Instance type")
    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = create_pipeline(role_arn=args.role_arn, pipeline_run_uuid=args.pipeline_run_id)
    pipeline.upsert(role_arn=args.role_arn)

    execution = pipeline.start(
        parameters={
            "PipelineRunID": args.pipeline_run_id,
            "ImageURI": args.image_uri,
            "InferenceImageURI": args.inference_image_uri,
            # "RoleARN": args.role_arn,
            # "JobPrefixName": "Text2Cypher",
            "Environment": args.env,
            "WandbApiKey": args.wandb_api_key,
            "OpenAIApiKey": args.openai_api_key,
            "InputDataFolderURI": "s3://bl-portfolio-ml-sagemaker-source-data/notechat-dataset/",
            "PreprocessedOutputS3Uri": f"s3://bl-portfolio-ml-sagemaker-{args.env}/input/preprocessed",
            "TrainingOutputS3Uri": f"s3://bl-portfolio-ml-sagemaker-{args.env}/output/artifacts",
            "PackagedModelS3Uri": f"s3://bl-portfolio-ml-sagemaker-{args.env}/output/artifacts/{args.pipeline_run_id}/model.tar.gz",
            "PreprocessingInstanceType": args.preprocessing_instance_type,
            "PreprocessingInstanceCount": args.preprocessing_instance_count,
            "TrainingInstanceType": args.training_instance_type,
            "TrainingInstanceCount": args.training_instance_count,
            "EvaluationInstanceType": args.evaluation_instance_type,
            "EvaluationInstanceCount": args.evaluation_instance_count,
            "DeploymentInstanceType": args.deployment_instance_type
        }
    )
    execution.wait(delay=60, max_attempts=240)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        exit(1)
