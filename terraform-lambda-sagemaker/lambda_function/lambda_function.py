import boto3
import uuid

def lambda_handler(event, context):
    sagemaker_client = boto3.client('sagemaker')

    model_package_arn = event['model_package_arn']
    endpoint_name = event['endpoint_name']
    instance_type = event['instance_type']
    role = event['role']

    # Generate a unique model name for this deployment
    model_name = f"{endpoint_name}-{str(uuid.uuid4())[:8]}"
    endpoint_config_name = f"{endpoint_name}-config"

    # Create model from registered package
    sagemaker_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        PrimaryContainer={
            'ModelPackageName': model_package_arn
        }
    )

    # Create endpoint config
    sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InitialInstanceCount': 1,
            'InstanceType': instance_type,
        }]
    )

    # Deploy endpoint
    sagemaker_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )

    return {
        'status': 'Deployment triggered',
        'endpoint_name': endpoint_name
    }
