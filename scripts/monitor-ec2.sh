#!/bin/bash

# NBA Inference EC2 Monitoring Script
# Usage: ./monitor-ec2.sh <instance-ip>

set -e

INSTANCE_IP=${1:-""}
ENVIRONMENT=${2:-"dev"}

if [ -z "$INSTANCE_IP" ]; then
    echo "Usage: $0 <instance-ip> [environment]"
    echo "Example: $0 54.123.45.67 dev"
    exit 1
fi

echo "🔍 Monitoring NBA Inference Service on $INSTANCE_IP"
echo "Environment: $ENVIRONMENT"
echo "Timestamp: $(date)"
echo ""

# Function to check endpoint
check_endpoint() {
    local endpoint=$1
    local description=$2
    
    echo -n "Checking $description... "
    if curl -f -s --max-time 10 "$endpoint" > /dev/null 2>&1; then
        echo "✅ OK"
        return 0
    else
        echo "❌ FAILED"
        return 1
    fi
}

# Function to get response time
get_response_time() {
    local endpoint=$1
    local time=$(curl -o /dev/null -s -w '%{time_total}' --max-time 10 "$endpoint" 2>/dev/null || echo "timeout")
    echo "$time"
}

# Health check
echo "🏥 Health Check:"
check_endpoint "http://$INSTANCE_IP:8000/health" "Health endpoint"

# API documentation
echo ""
echo "📚 API Documentation:"
check_endpoint "http://$INSTANCE_IP:8000/docs" "API docs"

# Test inference
echo ""
echo "🧠 Inference Test:"
echo -n "Testing inference endpoint... "
response=$(curl -s --max-time 30 -X POST "http://$INSTANCE_IP:8000/summarize_recap" \
  -H "Content-Type: application/json" \
  -d '{
    "game_recap": "Lakers beat Suns in overtime with a spectacular game-winning shot by Bryant. The game was intense with both teams playing at their best.",
    "max_length": 50
  }' 2>/dev/null || echo "timeout")

if echo "$response" | grep -q "game_recap_summary"; then
    echo "✅ OK"
    summary=$(echo "$response" | jq -r '.game_recap_summary' 2>/dev/null || echo "Could not parse response")
    echo "   Generated summary: $summary"
else
    echo "❌ FAILED"
    echo "   Response: $response"
fi

# Performance metrics
echo ""
echo "⚡ Performance Metrics:"
health_time=$(get_response_time "http://$INSTANCE_IP:8000/health")
echo "   Health check response time: ${health_time}s"

# System information (if SSH is available)
echo ""
echo "🖥️  System Information:"
echo "   Instance IP: $INSTANCE_IP"
echo "   Environment: $ENVIRONMENT"
echo "   Monitoring time: $(date)"

# Service status check via SSH (optional)
if command -v ssh >/dev/null 2>&1; then
    echo ""
    echo "🔧 Service Status (via SSH):"
    echo "   To check service status manually:"
    echo "   ssh -i your-key.pem ec2-user@$INSTANCE_IP 'sudo systemctl status nba-inference'"
    echo "   ssh -i your-key.pem ec2-user@$INSTANCE_IP 'sudo docker-compose -f /opt/inference/docker-compose.yml ps'"
fi

echo ""
echo "📊 Monitoring complete!"
echo "   Health: http://$INSTANCE_IP:8000/health"
echo "   API Docs: http://$INSTANCE_IP:8000/docs"
echo "   Inference: http://$INSTANCE_IP:8000/summarize_recap"
