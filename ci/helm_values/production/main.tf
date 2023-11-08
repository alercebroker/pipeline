provider "aws" {
  region = "us-east-1"
}

terraform {
  backend "s3" {
    bucket         = "alerce-infrastructure-terraform-state"
    key            = "state/helm_values/production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    kms_key_id     = "alias/terraform-bucket-key"
    dynamodb_table = "terraform-state"
  }
}

data "aws_msk_cluster" "msk_elasticc" {
  cluster_name = "internal-elasticc"
}
data "aws_msk_cluster" "msk_public" {
  cluster_name = "public-production"
}

resource "aws_ssm_parameter" "correction_step_elasticc" {
  name = "correction_step-helm-values-elasticc"
  type = "String"
  value = templatefile("templates/correction_step_helm_values_elasticc.tftpl", {
    kafka_server   = data.aws_msk_cluster.msk_elasticc.bootstrap_brokers_sasl_scram
    kafka_username = var.correction_kafka_username
    kafka_password = var.correction_kafka_password
    ghcr_username  = var.ghcr_username
    ghcr_password  = var.ghcr_password
  })
}

resource "aws_ssm_parameter" "feature_step_elasticc" {
  name = "feature_step-helm-values-elasticc"
  type = "String"
  value = templatefile("templates/feature_step_helm_values_elasticc.tftpl", {
    feature_version = "6"
    kafka_server    = data.aws_msk_cluster.msk_elasticc.bootstrap_brokers_sasl_scram
    kafka_username  = var.feature_kafka_username
    kafka_password  = var.feature_kafka_password
    ghcr_username   = var.ghcr_username
    ghcr_password   = var.ghcr_password
  })
}

resource "aws_ssm_parameter" "balto_step_elasticc" {
  name = "balto-helm-values"
  type = "String"
  value = templatefile("templates/balto_helm_values.tftpl", {
    kafka_server        = data.aws_msk_cluster.msk_elasticc.bootstrap_brokers_sasl_scram
    kafka_public_server = data.aws_msk_cluster.msk_public.bootstrap_brokers_sasl_scram
    kafka_username      = var.balto_kafka_username
    kafka_password      = var.balto_kafka_password
    ghcr_username       = var.ghcr_username
    ghcr_password       = var.ghcr_password
    model_path          = var.balto_model_path
    quantiles_path      = var.balto_quantiles_path
  })
}

resource "aws_ssm_parameter" "messi_step_elasticc" {
  name = "messi-helm-values"
  type = "String"
  value = templatefile("templates/messi_helm_values.tftpl", {
    kafka_server            = data.aws_msk_cluster.msk_elasticc.bootstrap_brokers_sasl_scram
    kafka_public_server     = data.aws_msk_cluster.msk_public.bootstrap_brokers_sasl_scram
    kafka_username          = var.messi_kafka_username
    kafka_password          = var.messi_kafka_password
    ghcr_username           = var.ghcr_username
    ghcr_password           = var.ghcr_password
    model_path              = var.messi_model_path
    quantiles_path          = var.balto_quantiles_path
    features_quantiles_path = var.messi_quantiles_path
  })
}

resource "aws_ssm_parameter" "tinkywinky_step_elasticc" {
  name = "tinkywinky-helm-values"
  type = "String"
  value = templatefile("templates/tinkywinky_helm_values.tftpl", {
    kafka_server        = data.aws_msk_cluster.msk_elasticc.bootstrap_brokers_sasl_scram
    kafka_public_server = data.aws_msk_cluster.msk_public.bootstrap_brokers_sasl_scram
    kafka_username      = var.tinkywinky_kafka_username
    kafka_password      = var.tinkywinky_kafka_password
    ghcr_username       = var.ghcr_username
    ghcr_password       = var.ghcr_password
    model_path          = var.tinkywinky_model_path
  })
}

resource "aws_ssm_parameter" "lightcurve_step_elasticc" {
  name = "lightcurve-step-helm-values-elasticc"
  type = "String"
  value = templatefile("templates/lightcurve_step_helm_values_elasticc.tftpl", {
    kafka_server   = data.aws_msk_cluster.msk_elasticc.bootstrap_brokers_sasl_scram
    kafka_username = var.lightcurve_kafka_username
    kafka_password = var.lightcurve_kafka_password
    ghcr_username  = var.ghcr_username
    ghcr_password  = var.ghcr_password
  })
}

resource "aws_ssm_parameter" "magstats_step_elasticc" {
  name = "magstats_step-helm-values-elasticc"
  type = "String"
  value = templatefile("templates/magstats_step_helm_values_elasticc.tftpl", {
    kafka_server   = data.aws_msk_cluster.msk_elasticc.bootstrap_brokers_sasl_scram
    kafka_username = var.magstats_kafka_username
    kafka_password = var.magstats_kafka_password
    ghcr_username  = var.ghcr_username
    ghcr_password  = var.ghcr_password
  })
}

resource "aws_ssm_parameter" "prv_candidates_step_elasticc" {
  name = "prv_candidates_step-helm-values-elasticc"
  type = "String"
  value = templatefile("templates/prv_candidates_step_helm_values_elasticc.tftpl", {
    kafka_server   = data.aws_msk_cluster.msk_elasticc.bootstrap_brokers_sasl_scram
    kafka_username = var.prv_candidates_kafka_username
    kafka_password = var.prv_candidates_kafka_password
    ghcr_username  = var.ghcr_username
    ghcr_password  = var.ghcr_password
  })
}

resource "aws_ssm_parameter" "scribe_step_elasticc" {
  name = "scribe_step-helm-values-elasticc"
  type = "String"
  value = templatefile("templates/scribe_helm_values_elasticc.tftpl", {
    kafka_server   = data.aws_msk_cluster.msk_elasticc.bootstrap_brokers_sasl_scram
    kafka_username = var.scribe_kafka_username
    kafka_password = var.scribe_kafka_password
    ghcr_username  = var.ghcr_username
    ghcr_password  = var.ghcr_password
  })
}

resource "aws_ssm_parameter" "sorting_hat_step_elasticc" {
  name = "sorting_hat_step-helm-values-elasticc"
  type = "String"
  value = templatefile("templates/sorting_hat_step_helm_values_elasticc.tftpl", {
    kafka_server   = data.aws_msk_cluster.msk_elasticc.bootstrap_brokers_sasl_scram
    kafka_username = var.sorting_hat_kafka_username
    kafka_password = var.sorting_hat_kafka_password
    ghcr_username  = var.ghcr_username
    ghcr_password  = var.ghcr_password
  })
}

resource "aws_ssm_parameter" "xmatch_step_elasticc" {
  name = "xmatch_step-helm-values-elasticc"
  type = "String"
  value = templatefile("templates/xmatch_step_helm_values_elasticc.tftpl", {
    kafka_server   = data.aws_msk_cluster.msk_elasticc.bootstrap_brokers_sasl_scram
    kafka_username = var.xmatch_kafka_username
    kafka_password = var.xmatch_kafka_password
    ghcr_username  = var.ghcr_username
    ghcr_password  = var.ghcr_password
  })
}
