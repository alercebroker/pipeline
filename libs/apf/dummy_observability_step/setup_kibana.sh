#!/bin/bash
# Script to automatically create Kibana/OpenSearch Dashboards index pattern

KIBANA_URL="http://localhost:5601"
INDEX_PATTERN="dummy-step-logs"
TIME_FIELD="@timestamp"

echo "Waiting for Kibana to be ready..."
until curl -s "${KIBANA_URL}/api/status" | grep -q '"state":"green"'; do
    echo "Kibana not ready yet, waiting..."
    sleep 5
done

echo "Kibana is ready. Creating index pattern..."

# Create the index pattern using Kibana API
curl -X POST "${KIBANA_URL}/api/saved_objects/index-pattern/${INDEX_PATTERN}" \
  -H "osd-xsrf: true" \
  -H "Content-Type: application/json" \
  -d "{
    \"attributes\": {
      \"title\": \"${INDEX_PATTERN}\",
      \"timeFieldName\": \"${TIME_FIELD}\"
    }
  }" 2>/dev/null

# Set it as the default index pattern
curl -X POST "${KIBANA_URL}/api/opensearch-dashboards/settings/defaultIndex" \
  -H "osd-xsrf: true" \
  -H "Content-Type: application/json" \
  -d "{
    \"value\": \"${INDEX_PATTERN}\"
  }" 2>/dev/null

echo ""
echo "✅ Index pattern created successfully!"
echo "📊 View logs at: ${KIBANA_URL}/app/discover#/"
