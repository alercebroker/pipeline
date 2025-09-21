# ALeRCE Pipeline Observability Context

## Overview

The ALeRCE pipeline processes astronomical alerts using a series of containerized steps deployed in Kubernetes. Each step consumes and produces messages (usually via Kafka topics), and performs operations such as saving data to PostgreSQL, computing features, running ML models, etc. Observability (logs and metrics) is critical for monitoring, debugging, and optimizing pipeline behavior.

---

## Logging

- **All pipeline steps use structured Python logging**, emitting logs (info, debug, warning, error) to stdout/stderr.
- **Filebeat runs as a DaemonSet in Kubernetes**, collecting logs from all step containers.
- **Filebeat enriches logs with Kubernetes metadata** (pod name, namespace, node, container, labels).
- **Logs are shipped directly from Filebeat to OpenSearch** (Elasticsearch), enabling centralized log storage, search, filtering, and audit.
- **Kibana dashboards** are used for log analytics, troubleshooting, and alerting (errors, discards, exceptions).
- **Log format is structured (JSON preferred)** for easy filtering and correlation with pipeline context (e.g., batch ID, object ID, alert ID, step name, discard reason).

---

## Metrics

- **Prometheus is used for operational metrics.**
  - Each step exposes Prometheus metrics via an HTTP `/metrics` endpoint (using the `prometheus_client` Python library).
  - **Metrics collected include:** consumer/arrival rate, processing rate, batch size, discarded message counts (by cause), processing times, lightcurve statistics, etc.
  - **Custom counters and histograms** track discards by cause, batch size, lightcurve length, and other statistics relevant to pipeline health and science.
  - Prometheus scrapes these endpoints at regular intervals and stores the metrics.
  - **Grafana dashboards** visualize step-level metrics, rates, timings, and discard statistics.
  - Alerts can be configured in Grafana for anomalous rates, errors, or other operational issues.

---

## Best Practices

- **Logs and metrics are separated:** logs go to OpenSearch/Kibana, metrics go to Prometheus/Grafana.
- **Structured logs and standardized metric names** for efficient search, aggregation, and cross-component analysis.
- **Log retention and compression** are configured in OpenSearch and in transit.
- **Kubernetes metadata enrichment** enables correlation between pipeline steps, nodes, and operational context.
- **Prometheus is the single source of truth for operational metrics** used in Grafana dashboards and alerts.

---

## Summary Table

| Layer       | Tool/Service            | Purpose                        |
|-------------|-------------------------|--------------------------------|
| Metrics     | Prometheus, Grafana     | Rates, timings, discard stats  |
| Logs        | Filebeat, OpenSearch, Kibana | Centralized logs, error audit, search |
| Enrichment  | Filebeat                | Add Kubernetes pod/node/labels to logs |
| Alerting    | Grafana, Kibana         | Metric/log-based alerts        |

---

This observability plan provides scalable, reliable, and actionable monitoring for all pipeline steps and infrastructure, supporting both operational and scientific needs.