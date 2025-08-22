from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.model import Model
from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput
from sagemaker.lambda_helper import Lambda

def create_pipeline(role_arn: str, pipeline_run_uuid: str = None) -> Pipeline:
    session = PipelineSession()

    # Parameters
    pipeline_run_id_param = ParameterString(name="PipelineRunID", default_value="no-pipeline-id")
    source_data_folder_uri = ParameterString(name="InputDataFolderURI", default_value="s3://bl-portfolio-ml-sagemaker-source-data/notechat-dataset/")
    # job_prefix_name = ParameterString(name="JobPrefixName", default_value="Project")
    env_param = ParameterString(name="Environment", default_value="dev")
    wandb_api_key = ParameterString(name="WandbApiKey", default_value="")
    open_ai_key = ParameterString(name="OpenAIApiKey", default_value="")
    image_uri = ParameterString(name="ImageURI", default_value="")
    inference_image_uri = ParameterString(name="InferenceImageURI", default_value="")
    preprocessing_instance_type = ParameterString(name="PreprocessingInstanceType", default_value="ml.m5.large")
    preprocessing_instance_count = ParameterInteger(name="PreprocessingInstanceCount", default_value=1)
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.large")
    training_instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=1)
    evaluation_instance_type = ParameterString(name="EvaluationInstanceType", default_value="ml.m5.large")
    evaluation_instance_count = ParameterInteger(name="EvaluationInstanceCount", default_value=1)
    deployment_instance_type = ParameterString(name="DeploymentInstanceType", default_value="ml.m5.large")
    project_config = ParameterString(name="ProjectConfig", default_value="config.dev")

    preprocessed_data_output_uri = ParameterString("PreprocessedOutputS3Uri", default_value="s3://bl-portfolio-ml-sagemaker-dev/input/preprocessed")
    training_artifacts_output_uri = ParameterString("TrainingOutputS3Uri", default_value="s3://bl-portfolio-ml-sagemaker-dev/output/artifacts")
    package_model_uri = ParameterString("PackagedModelS3Uri", default_value="s3://bl-portfolio-ml-sagemaker-dev/output/artifacts/no_pipeline_id/model.tar.gz")

    # Preprocessing
    preprocessing_processor = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        role=role_arn,
        instance_count=preprocessing_instance_count,
        instance_type=preprocessing_instance_type,
        volume_size_in_gb=30,
        env={
            "ENV": env_param,
            "WANDB_API_KEY": wandb_api_key,
            "PIPELINE_RUN_ID": pipeline_run_id_param,
        },
    )

    preprocessing_step = ProcessingStep(
        name="DataPreProcessing",
        processor=preprocessing_processor,
        code="scripts/preprocessing.py",
        job_arguments=[
            "--config-path", "src/text2cypher/finetuning/config",
            "--config-name", project_config
        ],
        inputs=[ProcessingInput(source=source_data_folder_uri, destination="/opt/ml/processing/input/source-data", input_name="source-data")],
        outputs=[ProcessingOutput(source="/opt/ml/processing/output/preprocessed", destination=preprocessed_data_output_uri, output_name="training-data")]
    )

    # Training
    training_processor = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        role=role_arn,
        instance_count=training_instance_count,
        instance_type=training_instance_type,
        volume_size_in_gb=30,
        env={
            "ENV": env_param,
            "WANDB_API_KEY": wandb_api_key,
            "PIPELINE_RUN_ID": pipeline_run_id_param,
        },
    )

    training_step = ProcessingStep(
        name="ModelTraining",
        processor=training_processor,
        code="scripts/train.py",
        job_arguments=[
            "--config-path", "src/text2cypher/finetuning/config",
            "--config-name", project_config
        ],
        inputs=[ProcessingInput(
            source=preprocessing_step.properties.ProcessingOutputConfig.Outputs["training-data"].S3Output.S3Uri,
            destination="/opt/ml/processing/input/preprocessed",
            input_name="training-data"
        )],
        outputs=[ProcessingOutput(source="/opt/ml/processing/output/model-artifacts", destination=training_artifacts_output_uri, output_name="model-artifacts")]
    )

    # Evaluation
    evaluation_processor = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        role=role_arn,
        instance_count=evaluation_instance_count,
        instance_type=evaluation_instance_type,
        volume_size_in_gb=30,
        env={
            "ENV": env_param,
            "WANDB_API_KEY": wandb_api_key,
            "OPENAI_API_KEY": open_ai_key,
            "PIPELINE_RUN_ID": pipeline_run_id_param,
        },
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation-metrics",
        path=f"{pipeline_run_uuid}/reports/eval_metrics.json"
    )

    evaluation_step = ProcessingStep(
        name="ModelEvaluation",
        processor=evaluation_processor,
        code="scripts/evaluate_model.py",
        job_arguments=[
            "--config-path", "src/text2cypher/finetuning/config",
            "--config-name", project_config
        ],
        inputs=[
            ProcessingInput(
                source=preprocessing_step.properties.ProcessingOutputConfig.Outputs["training-data"].S3Output.S3Uri,
                destination="/opt/ml/processing/input/preprocessed",
                input_name="training-data"
            ),
            ProcessingInput(
                source=training_step.properties.ProcessingOutputConfig.Outputs["model-artifacts"].S3Output.S3Uri,
                destination="/opt/ml/processing/input/model-artifacts",
                input_name="model-artifacts"
            )
        ],
        outputs=[ProcessingOutput(
            source="/opt/ml/processing/output/model-artifacts",
            destination=training_artifacts_output_uri,
            output_name="evaluation-metrics",
        )],
        property_files=[evaluation_report],
    )

    model = Model(
        image_uri=inference_image_uri,
        model_data=package_model_uri,
        role=role_arn,
        sagemaker_session=session,
    )

    register_model_step = ModelStep(
        name="RegisterNoteChatModel",
        step_args=model.register(
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=[deployment_instance_type],
            transform_instances=[deployment_instance_type],
            model_package_group_name="NoteChatModel",
            approval_status="Approved",
            description="Registered model for notechat generation",
            customer_metadata_properties={
                "pipeline_run_id": pipeline_run_id_param,
                "env": env_param,
            },
        ),
    )

    registered_model_package = register_model_step.properties.ModelPackageArn



    condition_step = ConditionStep(
        name="CheckBertScoreCondition",
        conditions=[ConditionGreaterThanOrEqualTo(
            left=JsonGet(step_name=evaluation_step.name, property_file=evaluation_report, json_path="bert_score"),
            right=0.8,
        )],
        if_steps=[register_model_step],
        else_steps=[],
    )

    return Pipeline(
        name="NoteChatPipeline",
        parameters=[
            source_data_folder_uri,
            preprocessed_data_output_uri,
            training_artifacts_output_uri,
            package_model_uri,
            pipeline_run_id_param,
            # job_prefix_name,
            env_param,
            wandb_api_key,
            open_ai_key,
            image_uri,
            inference_image_uri,
            preprocessing_instance_type,
            preprocessing_instance_count,
            training_instance_type,
            training_instance_count,
            evaluation_instance_type,
            evaluation_instance_count,
            deployment_instance_type,
            project_config,
        ],
        steps=[preprocessing_step, training_step, evaluation_step, condition_step],
    )
