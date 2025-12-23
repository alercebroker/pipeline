#!/bin/bash

# Script to run tests for all simulated data classes
# Each class has its own subdirectory in simulated_jsons/

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "Running tests for all SIMWV5 classes"
echo "========================================"
echo ""

# Array of all class directories
CLASSES=(
    "AGN_WFD_20251010"
    "CaRT_WFD_20251010"
    "EB_WFD_20251010"
    "ILOT_WFD_20251010"
    "KN-BULLA19_WFD_20251010"
    "KN-K17_WFD_20251010"
    "M-dwarf_WFD_20251010"
    "Mira_WFD_20251010"
    "PISN_WFD_20251010"
    "RRL_WFD_20251010"
    "SLSN-I_WFD_20251010"
    "SNII-NMF_WFD_20251010"
    "SNII-templates_WFD_20251010"
    "SNIIn-MOSFIT_WFD_20251010"
    "SNIa-91bg_WFD_20251010"
    "SNIa_WFD_20251010"
    "SNIax_WFD_20251010"
    "SNIb-templates_WFD_20251010"
    "SNIc-templates_WFD_20251010"
    "TDE_WFD_20251010"
    "uLens-Binary_WFD_20251010"
    "uLens-Single-GenLens_WFD_20251010"
    "uLens-Single-PyLIMA_WFD_20251010"
)

# Counter for success/failures
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_CLASSES=()

# Run tests for each class
for CLASS in "${CLASSES[@]}"; do
    echo -e "${YELLOW}Testing class: $CLASS${NC}"
    echo "----------------------------------------"
    
    JSONS_DIR="simulated_jsons/$CLASS" poetry run pytest -s tests/unittest/test_step_lsst.py
    TEST_RESULT=$?
    
    # Handle feature_extraction_timing folder
    if [ -d "feature_extraction_timing" ]; then
        echo "Moving timing files to timings/$CLASS..."
        
        # Create timings subdirectory for this class
        mkdir -p "timings/$CLASS"
        
        # Move all JSON files from feature_extraction_timing to timings/CLASS
        if [ -n "$(ls -A feature_extraction_timing/*.json 2>/dev/null)" ]; then
            mv feature_extraction_timing/*.json "timings/$CLASS/"
        fi
        
        # Remove the feature_extraction_timing folder
        rm -rf feature_extraction_timing
        echo "Timing files moved and cleaned up."
    fi
    
    if [ $TEST_RESULT -eq 0 ]; then
        echo -e "${GREEN}✓ $CLASS: PASSED${NC}"
        ((SUCCESS_COUNT++))
    else
        echo -e "${RED}✗ $CLASS: FAILED${NC}"
        ((FAIL_COUNT++))
        FAILED_CLASSES+=("$CLASS")
    fi
    echo ""
done

# Summary
echo "========================================"
echo "SUMMARY"
echo "========================================"
echo -e "${GREEN}Passed: $SUCCESS_COUNT${NC}"
echo -e "${RED}Failed: $FAIL_COUNT${NC}"

if [ $FAIL_COUNT -gt 0 ]; then
    echo ""
    echo "Failed classes:"
    for FAILED_CLASS in "${FAILED_CLASSES[@]}"; do
        echo -e "  ${RED}- $FAILED_CLASS${NC}"
    done
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
