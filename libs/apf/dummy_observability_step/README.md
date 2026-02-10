# APF Observability Implementation Guide

Complete guide for implementing observability (structured logging and metrics) in APF pipeline steps.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Implementation Guide](#implementation-guide)
- [Configuration](#configuration)
- [Viewing Observability Data](#viewing-observability-data)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)

---

## Overview

Production-ready observability for APF pipeline steps with:
- **Structured JSON logging** - contextual fields for debugging
- **Prometheus metrics** - counters, histograms, gauges
- **Fluent Bit → OpenSearch → Kibana** - log aggregation and search
- **Grafana dashboards** - pre-configured metrics visualization

---

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- Conda (recommended) or virtualenv

### 1. Setup Environment

```bash
conda create -n apf_observability python=3.10 -y
conda activate apf_observability
cd /path/to/libs/apf && pip install -e .
```

### 2. Start Observability Stack

```bash
cd /path/to/dummy_observability_step
docker compose up -d
# Wait ~30s for services to start (OpenSearch:9200, Kibana:5601, Prometheus:9090, Grafana:3000)
docker compose ps  # Verify all healthy
```

### 3. Run the Example

```bash
python dummy_step_with_metrics.py
```

You'll see:
- JSON-formatted logs in stdout
- Metrics server starts on port 8000
- Processes 4 messages from dummy_data.csv
- Keeps metrics endpoint alive for 120 seconds

### Step 4: View Observability Data

# Processes 4 messages, exposes metrics on :8000 for 120 seconds
```

### 4. View Data

- **Kibana:** http://localhost:5601/app/discover (index auto-created)
- **Grafana:** http://localhost:3000 (admin/admin) → "APF Step Observability Dashboard"
- **Prometheus:** http://localhost:9090 → query `apf_messages_consum
│  ┌──────────────┐   │
│  │ JSONFormatter│   │──► JSON logs to stdout & file
│  └──────────────┘   │
│                     │
│  ┌──────────────┐   │
│  │ Prometheus   │   │──► Metrics on :8000/metrics
│  │   Metrics    │   │
│  └──────────────┘   │
└─────────────────────┘
         │
         ├──► Logs ──► Fluent Bit ──► OpenSearch ──► Kibana
         │            (enrichment)     (storage)     (visualization)
         │
         └──► Metrics ──► Prometheus ──► Grafana
                         (scraping)      (dashboards)
```

**Flow:**
1. Step writes JSON logs to `dummy_step.log` file
2. Fluent Bit tails the file with `Read_from_Head On` to capture all logs
3. Fluent Bit parses JSON and adds Kubernetes-like metadata
4. Logs indexed in OpenSearch, viewable in Kibana
5. Prometheus scrapes `/metrics` endpoint every 15 seconds
6. Metrics retained for 15 days (configurable)
7. Grafana queries Prometheus for dashboard visualization

---

## Implementation Guide

### 1. Structured Logging

#### JSON Formatter
→ file
2. Fluent Bit tails file → parses JSON → adds metadata → OpenSearch
3. Prometheus scrapes `/metrics` every 15s → retains 15 days
4. Kibana/Grafana query their respective backends for
    """Format logs as structured JSON with custom fields"""
    
    def format(self, record):
        log_record = {
            "level": record.levelname,
            "timestamp": self.formatTime(record, self.datefmt),
            "step": record.name,
            "message": record.getMessage(),
            "survey": getattr(record, "survey", "unknown"),
        }
        
        # Add extra context fields if present
        for field in ["n_messages", "processing_time_ms", "oid", "aid", 
                      "discard_reason", "error_type", "batch_id"]:
            if hasattr(record, field):
                log_record[field] = getattr(record, field)
        
        # Add exception traceback if present
        if record.exc_info:
            log_record["traceback"] = self.formatException(record.exc_info)
        
        return json.dumps(log_record)
```

#### Step Integration

```python
from apf.core.step import GenericStep

class MyStep(GenericStep):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.survey = config.get("SURVEY", "unknown")
        
        # Configure JSON logging to stdout
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        self.logger.handlers = [handler]
        self.logger.setLevel(config.get("LOGGING_LEVEL", "INFO"))
        
        # Optional: Add file handler for Fluent Bit
        file_handler = logging.FileHandler("my_step.log")
        file_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(file_handler)
    
    def execute(self, messages):
        # Log with context
        self.logger.info(
            "Processing batch started",
            extra={
                "n_messages": len(messages),
                "survey": self.survey
            }
        )
        
        for msg in messages:
            try:
                result = self.process(msg)
                self.logger.info(
                    "Message processed",
                    extra={
                        "oid": msg.get("oid"),
                        "aid": msg.get("aid")
                    }
                )
            except Exception as e:
                self.logger.error(
                    "Processing failed",
                    extra={
                        "oid": msg.get("oid"),
                        "error_type": type(e).__name__
                    },
                    exc_info=True
                )
        
        return messages
```

#### Logging Best Practices

**Use Appropriate Log Levels:**
- `DEBUG` - Verbose diagnostic information (disabled in production)
- `INFO` - Normal operation events (batch processed, message received)
- Log Levels:** DEBUG (verbose) → INFO (normal ops) → WARNING (unexpected) → ERROR (handled failures) → CRITICAL (serious)

**Always use structured fields:**
```python
# ✅ GOOD
self.logger.info("Batch processed", extra={"n_messages": 10, "duration_ms": 250})
# ❌ BAD
self.logger.info(f"Processed 10 messages in 250ms")
```

**Essential fields:** step, survey, level, timestamp, message, n_messages, processing_time_ms, oid/aid (traceability), discard_reason, error_type, traceback
#### Metrics Setup

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

class MyStep(GenericStep):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        
        # Start metrics HTTP server
        metrics_port = config.get("METRICS_PORT", 8000)
        start_http_server(metrics_port)
        self.logger.info("Prometheus metrics server started", 
                        extra={"port": metrics_port})
        
        # Define metrics
        self.messages_consumed = Counter(
            'apf_messages_consumed_total',
            'Total number of messages consumed from input',
            ['step', 'survey']
        )
        
        self.messages_processed = Counter(
            'apf_messages_processed_total',
            'Total number of messages successfully processed',
            ['step', 'survey']
        )
        
        self.messages_discarded = Counter(
            'apf_messages_discarded_total',
            'Total number of messages discarded',
            ['step', 'survey', 'reason']
        )
        
        self.processing_time = Histogram(
            'apf_processing_time_seconds',
            'Time spent processing message batches',
            ['step', 'survey'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
        
        self.batch_size = Histogram(
            'apf_batch_size_messages',
            'Distribution of message batch sizes',
            ['step', 'survey'],
            buckets=[1, 5, 10, 50, 100, 500, 1000]
        )
    
    def execute(self, messages):
        step_name = self.__class__.__name__
        
        # Record batch size
        self.batch_size.labels(step=step_name, survey=self.survey).observe(len(messages))
        
        # Time the processing
        start_time = time.time()
        
        for msg in messages:
            # Count consumed message
            self.messages_consumed.labels(step=step_name, survey=self.survey).inc()
            
            # Validate and discard if needed
            if not self.is_valid(msg):
                self.messages_discarded.labels(
                    step=step_name,
                    survey=self.survey,
                    reason="invalid_data"
                ).inc()
                continue
            
            # Process and count success
            self.process(msg)
            self.messages_processed.labels(step=step_name, survey=self.survey).inc()
        
        # Record processing time
        duration = time.time() - start_time
        self.processing_time.labels(step=step_name, survey=self.survey).observe(duration)
        
        return messages
```

#### Metric Types

**Counters** (monotonically increasing):
```python
self.counter = Counter('apf_events_total', 'Description', ['step', 'survey'])
self.counter.labels(step="MyStep", survey="ZTF").inc()  # Increment by 1
self.counter.labels(step="MyStep", survey="ZTF").inc(5)  # Increment by 5
```

**Histograms** (distributions with quantiles):
```python
self.histogram = Histogram('apf_duration_seconds', 'Description', ['step'])
self.histogram.labels(step="MyStep").observe(0.523)  # Record observation
```

**Gauges** (current value that can go up or down):
```python
self.gauge = Gauge('apf_active_tasks', 'Description', ['step'])
self.gauge.labels(step="MyStep").inc()   # Increment
- **Counter** (always increasing): `.inc()` or `.inc(n)`
- **Histogram** (distributions): `.observe(value)` - auto-calculates quantiles
- **Gauge** (current value): `.inc()`, `.dec()`, `.set(value)Gauges:**
- `apf_active_processing_batches{step}` - Currently processing
- `apf_queue_depth_messages{step, survey}` - Backlog size
- `apf_active_connections{step, service}` - Active DB/API connections

#### Metrics Best Practices

✅ **DO:**
- Use consistent label names across all steps
- Keep label cardinality low (avoid UUIDs, timestamps as labels)
- Use descriptive metric names with units (e.g., `_seconds`, `_messages`)
- Document metrics with clear help text
- Use histograms for timing and size distributions

❌ **DON'T:**
- Create unbounded label values (e.g., user IDs, object IDs as labels)
- Use metrics for high-cardinality data (use logs instead)
- Mix metric types for the same metric name
- Change metric types after deployment

---

## Configuration

### Docker Compose Configuration

Key configuration in `docker-compose.yml`:

```yaml
services:
  opensearch:
    image: opensearchproject/opensearch:2
    environment:
      - discovery.type=single-node
      - plugins.security.disabled=true
    ports:
      - 9200:9200

  kibana:
    image: opensearchproject/opensearch-dashboards:2
    ports:
      - 5601:5601
    depends_on:
      - opensearch
    healthcheck:
      test: ["CMD-SHELL", "curl -s http://localhost:5601/api/status | grep -q '\"state\":\"green\"'"]

  kibana-setup:
    image: curlimages/curl:latest
    depends_on:
      kibana:
        condition: service_healthy
    command:
      - -c
      - |
        curl -X POST 'http://kibana:5601/api/saved_objects/index-pattern/dummy-step-logs' \
          -H 'osd-xsrf: true' \
          -H 'Content-Type: application/json' \
    Key Settings

**docker-compose.yml:**
- `kibana-setup` auto-creates index pattern on startup
- Prometheus/Grafana use `network_mode: host` to access step on `localhost:8000`

**fluent-bit.conf:**
- `Read_from_Head On` - critical to read entire log file
- `Suppress_Type_Name On` - required for OpenSearch 2.x compatibility

**prometheus.yml:**
- Scrape interval: 15s
- Target: `localhost:8000`
- Retention: 15 days (default)

**Environment Variables:**
- `LOGGING_LEVEL` - DEBUG, INFO, WARNING, ERROR, CRITICAL
- `METRICS_PORT` - Default 8000
- `SURVEY` - Survey identifier (ZTF, LSST, ATLAS)
- `POD_NAME`, `POD_NAMESPACE` - Kubernetes metadata (auto-injected in prod) Prometheus (Metrics)

**URL:** http://localhost:9090

**Example Queries:**
```promql
# Current values
apf_messages_consumed_total
apf_messages_processed_total{exported_step="DummyStepWithMetrics"}

# Rates (messages per second)
rate(apf_messages_consumed_total[5m])
rate(apf_messages_processed_total[5m])

# Processing time percentiles
histogram_quantile(0.50, rate(apf_processing_time_seconds_bucket[5m]))  # median
histogram_quantile(0.95, rate(apf_processing_time_seconds_bucket[5m]))  # 95th percentile
histogram_quantile(0.99, rate(apf_processing_time_seconds_bucket[5m]))  # 99th percentile

# Discard rate
rate(apf_messages_discarded_total[5m])

# Average batch size
rate(apf_batch_size_messages_sum[5m]) / rate(apf_batch_size_messages_count[5m])
```

**Important Notes:**
- Instant queries show current values (only works when target is UP)
- Range queries show historical data (works even when target is DOWN)
- Data retention: 15 days by default
- Scrape interval: 15 seconds

### Grafana (Dashboards)

**URL:** http://localhost:3000
**Login:** admin / admin

**Pre-configured Dashboard:** "APF Step Observability Dashboard"

**Panels:**
1. **Message Processing Rate** - Messages/second over time
2. **Total Counters** - Consumed, processed, discarded counts
3. **Processing Time Percentiles** - p50, p95, p99 latency
4. **Batch Size Distribution** - Histogram of batch sizes
5. **Discard Rate** - Rate of discarded messages by reason
6. **Error Rate** - Failed messages over time
7. **Active Processing** - Current number of active batches (gauge)

**Dashboard Features:**
- Auto-refresh every 30 seconds
- Time range selector
- Variable filters for step and survey
- Annotations for deployments/incidents

---

## Production Deployment

### Kubernetes

**Logging:**
- Write JSON to stdout (no files)
- Fluent Bit DaemonSet auto-collects and enriches with K8s metadata
- Inject pod info: `POD_NAME`, `POD_NAMESPACE`, `NODE_NAME` via fieldRef

**Metrics:**
- Expose port 8000 in Service
- Use ServiceMonitor or annotations: `prometheus.io/scrape: "true"`, `prometheus.io/port: "8000"`

**Best Practices:**
```python
# Use env vars
log_level = os.getenv("LOGGING_LEVEL", "INFO")
metrics_port = int(os.getenv("METRICS_PORT", "8000"))

# Add K8s metadata
log_record["kubernetes_pod_name"] = os.getenv("POD_NAME", "unknown")

# Graceful shutdown
def tear_down(self):
    for handler in self.logger.handlers:
        handler.flush()
    self.active_tasks.labels(step=self.name).set(0)

# Health check
self.health = Gauge('apf_health_status', 'Step health', ['step'])
self.health.labels(step=self.name).set(1)

# 4. Query Prometheus API
curl http://localhost:9090/api/v1/query?query=apf_messages_consumed_total
```

**Common Fixes:**
- **Target down:** Step finished (only runs 120 seconds). Restart: `python dummy_step_with_metrics.py`
- **Network issue:** Ensure Prometheus uses `network_mode: host` in docker-compose.yml
- **Wrong labels:** Check Grafana queries use `exported_step="DummyStepWithMetrics"` not `step=...`

#### 3. Grafana Shows "No Data"

**Symptoms:** Dashboard panels empty or show "No data"

**Checks:**
```bash
# 1. Verify Prometheus datasource
# Open: http://localhost:3000/connections/datasources
# Check Prometheus URL: http://localhost:9090

# 2. Test datasource connection
# Click "Test" button in datasource settings

# 3. Check Prometheus has data
curl http://localhost:9090/api/v1/query?query=apf_messages_consumed_total
```

**Common Fixes:**
- **Wrong datasource URL:** Update to `http://localhost:9090` if using host network
- **Wrong metric labels:** Update dashboard queries to use `exported_step` instead of `step`
- **Time range:** Adjust dashboard time range (top right) to "Last 15 minutes"
- **No data yet:** Run the Python step to generate metrics

#### 4. OpenSearch Index Mapping Issues

**Symptoms:** Fields not searchable or wrong type in Kibana

**Fixes:**
```bash
# Delete and recreate index
curl -X DELETE http://localhost:9200/dummy-step-logs

# Recreate Fluent Bit to re-index
docker compose rm -f fluent-bit
docker compose up -d fluent-bit

# Refresh index pattern in Kibana
# Management → Index Patterns → dummy-step-logs → Refresh field list
```

#### 5. Port Already in Use

**Symptoms:** `Address already in use` error when starting step

**Fixes:**
```bash
# Find process using port 8000
lsof -i :8000
# or
netstat -tulpn | grep 8000

# Kill the process
kill <PID>

# Or use a different port
export METRICS_PORT=8001
python dummy_step_with_metrics.py
```

### Debug Mode

Enable verbose logging for troubleshooting:

```python

**Logs not in Kibana:**
```bash
ls -lh dummy_step.log && tail -20 dummy_step.log  # Verify file exists
docker logs fluent-bit  # Check for "initializing"
curl http://localhost:9200/dummy-step-logs/_count  # Verify index
# Fix: rm -rf dummy_step.log (if it's a directory)
# Fix: docker compose rm -f fluent-bit && docker compose up -d fluent-bit
```

**Metrics not in Prometheus:**
```bash
ps aux | grep dummy_step_with_metrics  # Check step running
curl http://localhost:8000/metrics  # Test endpoint
# Fix: Restart step (runs only 120s)
# Fix: Check network_mode: host in docker-compose.yml
# Fix: Use exported_step="..." not step="..." in queries
```

**Grafana "No Data":**
```bash
curl http://localhost:9090/api/v1/query?query=apf_messages_consumed_total
# Fix: Update datasource URL to http://localhost:9090
# Fix: Adjust time range to "Last 15 minutes"
# Fix: Update queries to use exported_step label
```

**Port in use:**
```bash
lsof -i :8000 | grep -v COMMAND | awk '{print $2}' | xargs kill
# Or: export METRICS_PORT=8001
```

**Debug mode:** `export LOGGING_LEVEL=DEBUG**OpenSearch:** https://opensearch.org/docs/
- **Prometheus:** https://prometheus.io/docs/
- **Grafana:** https://grafana.com/docs/
- **Prometheus Python Client:** https://github.com/prometheus/client_python

---

## Support

For questions or issues:
- Review this guide and troubleshooting section
- Check the example code in `dummy_step_with_metrics.py`
- Refer to the ALeRCE Pipeline Observability Plan
- Contact the ALeRCE team

---

**Last Updated:** February 2026
**Status:** ✅ Tested and verified end-to-end


**Custom domain metrics:** Add histograms for lightcurve detections, feature computation time, DB query counts with custom buckets.

**Log sampling:** For high-volume steps, log only 1% of individual messages, always log batch summaries.

**Grafana alerts:** Configure for error rate spikes (`rate(apf_messages_failed_total[5m]) > 10`) and latency degradation (`histogram_quantile(0.95, ...) > 5`).