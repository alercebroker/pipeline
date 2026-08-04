import boto3
from botocore.exceptions import ClientError


def get_secret(secret_name: str, key_to_return="SecretString", region_name="us-east-1"):
    client = boto3.client(service_name="secretsmanager", region_name=region_name)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    return get_secret_value_response[key_to_return]
