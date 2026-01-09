#!/bin/bash

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Orchestrator Status Monitor${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"

# Check if orchestrator is running
ORCH_PID=$(ps aux | grep scheduled_orchestrator.sh | grep -v grep | awk '{print $2}')
if [ -z "$ORCH_PID" ]; then
    echo -e "${RED}✗ Orchestrator is NOT running${NC}\n"
else
    echo -e "${GREEN}✓ Orchestrator is RUNNING${NC} (PID: $ORCH_PID)\n"
fi

# Count active Claude agents
CLAUDE_COUNT=$(ps aux | grep "claude --dangerously" | grep -v grep | wc -l)
echo -e "Active Claude agents: ${YELLOW}${CLAUDE_COUNT}${NC}\n"

# Show recent log entries
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Recent Log Entries:${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"

LOGFILE="/home/gprice/projects/multiverse_dive/orchestrator_runs.log"
if [ -f "$LOGFILE" ]; then
    tail -30 "$LOGFILE"
else
    echo -e "${RED}Log file not found${NC}"
fi

echo -e "\n${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Git Status:${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"

cd /home/gprice/projects/multiverse_dive
git status --short | head -10

echo -e "\n${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Recent Commits:${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"

git log --oneline -5 2>/dev/null || echo "No commits yet"

echo -e "\n${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "Run ${YELLOW}./monitor_orchestrator.sh${NC} to refresh"
echo -e "View live logs: ${YELLOW}tail -f orchestrator_runs.log${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"
