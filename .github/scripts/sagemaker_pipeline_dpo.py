from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.workflow.properties import PropertyFile


def create_pipeline(role_arn: str, pipeline_run_uuid: str = None) -> Pipeline:
    session = PipelineSession()

    # Parameters
    pipeline_run_id_param = ParameterString(name="PipelineRunID", default_value="no-pipeline-id")
    source_data_folder_uri = ParameterString(name="InputDataFolderURI", default_value="s3://nba-recap-summarization-model-source-data/nba-recap-dataset/")
    env_param = ParameterString(name="Environment", default_value="dev")
    wandb_api_key = ParameterString(name="WandbApiKey", default_value="")
    open_ai_key = ParameterString(name="OpenAIApiKey", default_value="")
    hf_token = ParameterString(name="HFToken", default_value="")
    huggingfacehub_api_token = ParameterString(name="HuggingFaceHubApiToken", default_value="")
    image_uri = ParameterString(name="ImageURI", default_value="")
    inference_image_uri = ParameterString(name="InferenceImageURI", default_value="")
    preprocessing_instance_type = ParameterString(name="PreprocessingInstanceType", default_value="ml.m5.large")
    preprocessing_instance_count = ParameterInteger(name="PreprocessingInstanceCount", default_value=1)
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.large")
    training_instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=1)
    evaluation_instance_type = ParameterString(name="EvaluationInstanceType", default_value="ml.m5.large")
    evaluation_instance_count = ParameterInteger(name="EvaluationInstanceCount", default_value=1)
    deployment_instance_type = ParameterString(name="DeploymentInstanceType", default_value="ml.m5.large")
    project_config = ParameterString(name="ProjectConfig", default_value="config.dpo.staging")
    base_model_path = ParameterString(name="BaseModelPath", default_value="")

    preprocessed_data_output_uri = ParameterString("PreprocessedOutputS3Uri", default_value="s3://nba-recap-summarization-model-dev/input/preprocessed")
    training_artifacts_output_uri = ParameterString("TrainingOutputS3Uri", default_value="s3://nba-recap-summarization-model-dev/output/artifacts")

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
            "HF_TOKEN": hf_token,
            "HUGGINGFACEHUB_API_TOKEN": huggingfacehub_api_token,
            "PIPELINE_RUN_ID": pipeline_run_id_param,
        },
    )

    preprocessing_step = ProcessingStep(
        name="DPODataPreProcessing",
        processor=preprocessing_processor,
        code="scripts/dpo_preprocessing.py",
        job_arguments=[
            "--config-path", "src/nba_game_recap_summarizer/finetuning/config",
            "--config-name", project_config
        ],
        inputs=[ProcessingInput(source=source_data_folder_uri, destination="/opt/ml/processing/input/source-data", input_name="source-data")],
        outputs=[ProcessingOutput(source="/opt/ml/processing/output/preprocessed", destination=preprocessed_data_output_uri, output_name="training-data")]
    )

    # DPO Training
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
            "HF_TOKEN": hf_token,
            "HUGGINGFACEHUB_API_TOKEN": huggingfacehub_api_token,
            "PIPELINE_RUN_ID": pipeline_run_id_param,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "BASE_MODEL_DIR": "/opt/ml/processing/input/model-base",
        },
    )

    training_step = ProcessingStep(
        name="DPOTuning",
        processor=training_processor,
        code="scripts/dpo_tune.py",
        job_arguments=[
            "--config-path", "src/nba_game_recap_summarizer/finetuning/config",
            "--config-name", project_config
        ],
        inputs=[
            ProcessingInput(
                source=preprocessing_step.properties.ProcessingOutputConfig.Outputs["training-data"].S3Output.S3Uri,
                destination="/opt/ml/processing/input/preprocessed",
                input_name="training-data"
            ),
            ProcessingInput(
                source=base_model_path,
                destination="/opt/ml/processing/input/model-base",
                input_name="base-model"
            ),
        ],
        outputs=[ProcessingOutput(source="/opt/ml/processing/output/model-artifacts", destination=training_artifacts_output_uri, output_name="model-artifacts")]
    )

    # DPO Evaluation
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
            "HF_TOKEN": hf_token,
            "HUGGINGFACEHUB_API_TOKEN": huggingfacehub_api_token,
            "PIPELINE_RUN_ID": pipeline_run_id_param,
        },
    )

    # Property file to surface metrics
    evaluation_report = PropertyFile(
        name="DPOEvaluationReport",
        output_name="evaluation-metrics",
        path=f"{pipeline_run_uuid}/reports/eval_metrics.json"
    )

    evaluation_step = ProcessingStep(
        name="DPOEvaluation",
        processor=evaluation_processor,
        code="scripts/evaluate_dpo.py",
        job_arguments=[
            "--config-path", "src/nba_game_recap_summarizer/finetuning/config",
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

    return Pipeline(
        name="NBARecapDpoPipeline",
        parameters=[
            source_data_folder_uri,
            preprocessed_data_output_uri,
            training_artifacts_output_uri,
            pipeline_run_id_param,
            env_param,
            wandb_api_key,
            open_ai_key,
            hf_token,
            huggingfacehub_api_token,
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
            base_model_path,
        ],
        steps=[preprocessing_step, training_step, evaluation_step],
    )


