provider "aws" {
  region = "us-east-1"
}

terraform {
  backend "s3" {
    bucket         = "alerce-staging-terraform-state"
    key            = "state/helm_values/staging/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    kms_key_id     = "alias/terraform-bucket-key"
    dynamodb_table = "terraform-state"
  }
}

data "aws_msk_cluster" "msk_internal" {
  cluster_name = "internal-staging"
}
data "aws_opensearch_domain" "opensearch" {
  domain_name = "staging"
}

resource "aws_ssm_parameter" "logstash_step" {
  name      = "logstash-helm-values"
  overwrite = true
  type      = "String"
  value = templatefile("templates/logstash_helm_values.tftpl", {
    kafka_server           = data.aws_msk_cluster.msk_internal.bootstrap_brokers_sasl_scram
    kafka_username         = var.logstash_kafka_username
    kafka_password         = var.logstash_kafka_password
    elasticsearch_username = var.elasticsearch_username
    elasticsearch_password = var.elasticsearch_password
    elasticsearch_server   = "https://${data.aws_opensearch_domain.opensearch.endpoint}:443"
  })
}


resource "aws_ssm_parameter" "lightcurve_step" {
  name      = "lightcurve-step-helm-values"
  type      = "String"
  overwrite = true
  value = templatefile("templates/lightcurve_step_helm_values.tftpl", {
    kafka_server   = data.aws_msk_cluster.msk_internal.bootstrap_brokers_sasl_scram
    kafka_username = var.lightcurve_kafka_username
    kafka_password = var.lightcurve_kafka_password
    ghcr_username  = var.ghcr_username
    ghcr_password  = var.ghcr_password
  })
}
