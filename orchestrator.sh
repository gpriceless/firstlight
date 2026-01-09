#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Multiverse Dive Orchestrator - Autonomous Build${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"

# Phase 1: Work Agent
echo -e "${GREEN}[Phase 1/3]${NC} Launching Work Agent..."
echo -e "Task: Claim next unclaimed roadmap item and implement it\n"

claude --dangerously-skip-permissions --print "
Look at ROADMAP.md and identify the next unclaimed item that needs to be worked on.

First, search for any items marked as 'TODO', 'pending', or unclaimed. Start with the earliest group (Group A, then B, then C, etc.) and work through tracks in order.

Once you find the next item to work on:
1. Update ROADMAP.md to mark this item as 'IN PROGRESS' or 'WORKING'
2. Implement the item completely according to the specifications
3. Focus on writing clean, correct code that matches the architecture described in OPENSPEC.md
4. Do NOT write tests yet (that's for the next agent)
5. Do NOT commit or push (that's for the final agent)
6. When complete, simply finish - the next agent will review your work

Work efficiently and thoroughly. Make sure your implementation is complete before finishing.
"

if [ $? -ne 0 ]; then
    echo -e "${RED}Work Agent failed. Exiting.${NC}"
    exit 1
fi

echo -e "\n${GREEN}✓ Work Agent completed${NC}\n"

# Phase 2: Review and Test Agent
echo -e "${GREEN}[Phase 2/3]${NC} Launching Review & Test Agent..."
echo -e "Task: Review code, fix issues, and write comprehensive tests\n"

claude --dangerously-skip-permissions --print "
Review the code that was just implemented by the previous agent.

Your tasks:
1. Read through the newly added/modified code carefully
2. Check for any issues:
   - Logic errors
   - Style inconsistencies
   - Security vulnerabilities
   - Missing error handling
   - Performance issues
3. Fix any issues you find
4. Write comprehensive tests for the new functionality
   - Unit tests for individual functions
   - Integration tests if applicable
   - Edge case coverage
5. Run the tests to make sure they pass
6. Do NOT commit or push yet (that's for the final agent)

Make sure all tests pass before you finish.
"

if [ $? -ne 0 ]; then
    echo -e "${RED}Review Agent failed. Exiting.${NC}"
    exit 1
fi

echo -e "\n${GREEN}✓ Review & Test Agent completed${NC}\n"

# Phase 3: Documentation and Commit Agent
echo -e "${GREEN}[Phase 3/3]${NC} Launching Documentation & Commit Agent..."
echo -e "Task: Update documentation, commit changes, and push to GitHub\n"

claude --dangerously-skip-permissions --print "
Finalize the work that has been completed by the previous agents.

Your tasks:
1. Review what was implemented and tested
2. Update ROADMAP.md to mark the completed item as 'DONE' or 'COMPLETED'
3. Update any relevant documentation (README.md, OPENSPEC.md, docstrings, etc.)
4. Create a descriptive git commit with:
   - Clear commit message explaining what was implemented
   - Reference to the roadmap group/track
   - Summary of key changes
5. Push to GitHub

The commit message should be informative and follow this format:
'Implement [Feature Name] - [Group X, Track Y]

- Brief description of what was added
- Key components/files created
- Any important design decisions

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>'

Make sure the roadmap accurately reflects the completion status before pushing.
"

if [ $? -ne 0 ]; then
    echo -e "${RED}Documentation Agent failed. Exiting.${NC}"
    exit 1
fi

echo -e "\n${GREEN}✓ Documentation & Commit Agent completed${NC}\n"

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   ✨ Orchestrator Complete! ✨${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"
echo -e "Changes have been committed and pushed to GitHub."
echo -e "Check the repository to see the completed work!\n"
