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

data "aws_msk_cluster" "msk_staging" {
  cluster_name = "internal-staging"
}
data "aws_opensearch_domain" "opensearch" {
  domain_name = "staging"
}

data "aws_instance" "psql-alerts" {
  filter {
    name   = "tag:Name"
    values = ["psql-alerts-staging"]
  }
}
data "aws_instance" "psql-users" {
  filter {
    name   = "tag:Name"
    values = ["psql-users-staging"]
  }
}

resource "aws_ssm_parameter" "logstash_step" {
  name      = "logstash-helm-values"
  overwrite = true
  type      = "String"
  value = templatefile("templates/logstash_helm_values.tftpl", {
    kafka_server           = data.aws_msk_cluster.msk_staging.bootstrap_brokers_sasl_scram
    kafka_username         = var.logstash_kafka_username
    kafka_password         = var.logstash_kafka_password
    elasticsearch_username = var.elasticsearch_username
    elasticsearch_password = var.elasticsearch_password
    elasticsearch_server   = "https://${data.aws_opensearch_domain.opensearch.endpoint}:443"
  })
}

resource "aws_ssm_parameter" "s3_step" {
  name      = "s3_step-helm-values"
  type      = "String"
  overwrite = true
  value = templatefile("templates/s3_step_helm_values.tftpl", {
    kafka_server   = data.aws_msk_cluster.msk_staging.bootstrap_brokers_sasl_scram
    kafka_username = var.s3_kafka_username
    kafka_password = var.s3_kafka_password
    ghcr_username  = var.ghcr_username
    ghcr_password  = var.ghcr_password
  })
}

resource "aws_ssm_parameter" "early_classifier_step" {
  name      = "early_classifier_step-helm-values"
  type      = "String"
  overwrite = true
  value = templatefile("templates/early_classifier_step_helm_values.tftpl", {
    kafka_server   = data.aws_msk_cluster.msk_staging.bootstrap_brokers_sasl_scram
    kafka_username = var.early_classifier_kafka_username
    kafka_password = var.early_classifier_kafka_password
    ghcr_username  = var.ghcr_username
    ghcr_password  = var.ghcr_password
    db_host        = data.aws_instance.psql-alerts.private_ip
    db_username    = var.alerts_psql_username
    db_password    = var.alerts_psql_password
  })
}
resource "aws_ssm_parameter" "sorting_hat_step_ztf" {
  name      = "sorting_hat_step_ztf-helm-values"
  type      = "String"
  overwrite = true
  value = templatefile("templates/sorting_hat_step_ztf_helm_values.tftpl", {
    kafka_server   = data.aws_msk_cluster.msk_staging.bootstrap_brokers_sasl_scram
    kafka_username = var.sorting_hat_kafka_username
    kafka_password = var.sorting_hat_kafka_password
    ghcr_username  = var.ghcr_username
    ghcr_password  = var.ghcr_password
  })
}

resource "aws_ssm_parameter" "metadata_step" {
  name      = "metadata_step-helm-values"
  type      = "String"
  overwrite = true
  value = templatefile("templates/metadata_step_helm_values.tftpl", {
    kafka_server   = data.aws_msk_cluster.msk_staging.bootstrap_brokers_sasl_scram
    kafka_username = var.metadata_kafka_username
    kafka_password = var.metadata_kafka_password
    ghcr_username  = var.ghcr_username
    ghcr_password  = var.ghcr_password
  })
}

resource "aws_ssm_parameter" "prv_candidates_step" {
  name      = "prv_candidates_step-helm-values"
  type      = "String"
  overwrite = true
  value = templatefile("templates/prv_candidates_step_helm_values.tftpl", {
    kafka_server   = data.aws_msk_cluster.msk_staging.bootstrap_brokers_sasl_scram
    kafka_username = var.prv_candidates_kafka_username
    kafka_password = var.prv_candidates_kafka_password
    ghcr_username  = var.ghcr_username
    ghcr_password  = var.ghcr_password
  })
}

resource "aws_ssm_parameter" "lightcurve_step" {
  name      = "lightcurve-step-helm-values"
  type      = "String"
  overwrite = true
  value = templatefile("templates/lightcurve_step_helm_values.tftpl", {
    kafka_server   = data.aws_msk_cluster.msk_staging.bootstrap_brokers_sasl_scram
    kafka_username = var.lightcurve_kafka_username
    kafka_password = var.lightcurve_kafka_password
    ghcr_username  = var.ghcr_username
    ghcr_password  = var.ghcr_password
  })
}

resource "aws_ssm_parameter" "correction_step" {
  name      = "correction_step-helm-values"
  type      = "String"
  overwrite = true
  value = templatefile("templates/correction_step_helm_values.tftpl", {
    kafka_server   = data.aws_msk_cluster.msk_staging.bootstrap_brokers_sasl_scram
    kafka_username = var.correction_kafka_username
    kafka_password = var.correction_kafka_password
    ghcr_username  = var.ghcr_username
    ghcr_password  = var.ghcr_password
  })
}

resource "aws_ssm_parameter" "magstats_step" {
  name      = "magstats_step-helm-values"
  type      = "String"
  overwrite = true
  value = templatefile("templates/magstats_step_helm_values.tftpl", {
    kafka_server   = data.aws_msk_cluster.msk_staging.bootstrap_brokers_sasl_scram
    kafka_username = var.magstats_kafka_username
    kafka_password = var.magstats_kafka_password
    ghcr_username  = var.ghcr_username
    ghcr_password  = var.ghcr_password
  })
}

resource "aws_ssm_parameter" "xmatch_step" {
  name      = "xmatch_step-helm-values"
  type      = "String"
  overwrite = true
  value = templatefile("templates/xmatch_step_helm_values.tftpl", {
    kafka_server   = data.aws_msk_cluster.msk_staging.bootstrap_brokers_sasl_scram
    kafka_username = var.xmatch_kafka_username
    kafka_password = var.xmatch_kafka_password
    ghcr_username  = var.ghcr_username
    ghcr_password  = var.ghcr_password
  })
}

resource "aws_ssm_parameter" "features_step" {
  name      = "feature_step-helm-values"
  type      = "String"
  overwrite = true
  value = templatefile("templates/features_step_helm_values.tftpl", {
    kafka_server   = data.aws_msk_cluster.msk_staging.bootstrap_brokers_sasl_scram
    kafka_username = var.features_kafka_username
    kafka_password = var.features_kafka_password
    ghcr_username  = var.ghcr_username
    ghcr_password  = var.ghcr_password
  })
}

resource "aws_ssm_parameter" "lc_classification_step" {
  name      = "lc_classification_step-helm-values"
  type      = "String"
  overwrite = true
  value = templatefile("templates/lc_classification_step_helm_values.tftpl", {
    kafka_server        = data.aws_msk_cluster.msk_staging.bootstrap_brokers_sasl_scram
    kafka_public_server = data.aws_msk_cluster.msk_staging.bootstrap_brokers_sasl_scram
    kafka_username      = var.lc_classification_kafka_username
    kafka_password      = var.lc_classification_kafka_password
    ghcr_username       = var.ghcr_username
    ghcr_password       = var.ghcr_password
  })
}

resource "aws_ssm_parameter" "scribe_step_mongo" {
  name      = "scribe-helm-values"
  type      = "String"
  overwrite = true
  value = templatefile("templates/scribe_mongo_helm_values.tftpl", {
    namespace      = "scribe-step-mongo"
    db_type        = "mongo"
    group_id       = "scribe_consumer_mongo"
    kafka_server   = data.aws_msk_cluster.msk_staging.bootstrap_brokers_sasl_scram
    kafka_username = var.scribe_kafka_username
    kafka_password = var.scribe_kafka_password
    ghcr_username  = var.ghcr_username
    ghcr_password  = var.ghcr_password
  })
}

resource "aws_ssm_parameter" "scribe_step_sql" {
  name      = "scribe-psql-helm-values"
  type      = "String"
  overwrite = true
  value = templatefile("templates/scribe_psql_helm_values.tftpl", {
    namespace      = "scribe-step-sql"
    db_type        = "sql"
    group_id       = "scribe_consumer_psql"
    kafka_server   = data.aws_msk_cluster.msk_staging.bootstrap_brokers_sasl_scram
    kafka_username = var.scribe_kafka_username
    kafka_password = var.scribe_kafka_password
    ghcr_username  = var.ghcr_username
    ghcr_password  = var.ghcr_password
  })
}

resource "aws_ssm_parameter" "watchlist_step" {
  name      = "watchlist_step-helm-values"
  type      = "String"
  overwrite = true
  value = templatefile("templates/watchlist_step_helm_values.tftpl", {
    kafka_server       = data.aws_msk_cluster.msk_staging.bootstrap_brokers_sasl_scram
    kafka_username     = var.watchlist_kafka_username
    kafka_password     = var.watchlist_kafka_password
    ghcr_username      = var.ghcr_username
    ghcr_password      = var.ghcr_password
    alerts_db_host     = data.aws_instance.psql-alerts.private_ip
    users_db_host      = data.aws_instance.psql-users.private_ip
    alerts_db_password = var.alerts_psql_password
    alerts_db_username = var.alerts_psql_username
    users_db_password  = var.users_psql_password
    users_db_username  = var.users_psql_username
  })
}
