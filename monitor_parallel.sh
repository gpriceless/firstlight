#!/bin/bash

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

clear

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Parallel Orchestrator Status Monitor${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"

# Check if orchestrator is running
ORCH_PID=$(ps aux | grep scheduled_parallel_orchestrator.sh | grep -v grep | awk '{print $2}')
if [ -z "$ORCH_PID" ]; then
    echo -e "${YELLOW}⚠ Scheduled orchestrator is NOT running${NC}"
else
    echo -e "${GREEN}✓ Scheduled orchestrator is RUNNING${NC} (PID: $ORCH_PID)"
fi

# Count active Claude agents
CLAUDE_COUNT=$(ps aux | grep "claude --dangerously" | grep -v grep | wc -l)
echo -e "Active Claude agents: ${YELLOW}${CLAUDE_COUNT}${NC}"

# Count active parallel workers
PARALLEL_COUNT=$(ps aux | grep "PARALLEL.*AGENT" | grep -v grep | wc -l)
echo -e "Parallel workers: ${MAGENTA}${PARALLEL_COUNT}${NC}\n"

# Show track logs
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Track Logs:${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"

LOG_DIR="/home/gprice/projects/multiverse_dive/orchestrator_logs"
if [ -d "$LOG_DIR" ] && [ "$(ls -A $LOG_DIR 2>/dev/null)" ]; then
    for logfile in "$LOG_DIR"/*.log; do
        if [ -f "$logfile" ]; then
            filename=$(basename "$logfile")
            size=$(du -h "$logfile" | cut -f1)
            modified=$(stat -c '%y' "$logfile" | cut -d'.' -f1)

            echo -e "${CYAN}$filename${NC} (${size})"
            echo -e "  Last modified: ${modified}"

            # Show last few lines
            echo -e "  ${YELLOW}Latest:${NC}"
            tail -3 "$logfile" 2>/dev/null | sed 's/^/    /'
            echo ""
        fi
    done
else
    echo -e "${YELLOW}No track logs yet${NC}\n"
fi

# Show lock files (active claims)
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Active Track Claims:${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"

LOCK_DIR="/home/gprice/projects/multiverse_dive/.orchestrator_locks"
if [ -d "$LOCK_DIR" ] && [ "$(ls -A $LOCK_DIR 2>/dev/null)" ]; then
    for lockfile in "$LOCK_DIR"/*.lock; do
        if [ -f "$lockfile" ]; then
            filename=$(basename "$lockfile" .lock)
            pid=$(cat "$lockfile")
            if ps -p "$pid" > /dev/null 2>&1; then
                echo -e "${GREEN}✓${NC} ${filename} (PID: ${pid})"
            else
                echo -e "${RED}✗${NC} ${filename} (stale lock, PID ${pid} not running)"
            fi
        fi
    done
else
    echo -e "${YELLOW}No active claims${NC}"
fi

echo ""

# Git status
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Git Status:${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"

cd /home/gprice/projects/multiverse_dive
git status --short 2>/dev/null | head -15 || echo "No changes"

echo ""

# Recent commits
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Recent Commits:${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"

git log --oneline -5 2>/dev/null || echo "No commits yet"

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "Commands:"
echo -e "  ${YELLOW}./monitor_parallel.sh${NC} - Refresh this monitor"
echo -e "  ${YELLOW}tail -f orchestrator_logs/group_*_track_*.log${NC} - Watch specific track"
echo -e "  ${YELLOW}tail -f parallel_orchestrator_runs.log${NC} - Watch main log"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"
