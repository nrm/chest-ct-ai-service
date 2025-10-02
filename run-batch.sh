#!/bin/bash
# Batch processing runner for RadiAssist
# Usage: ./run-batch.sh /path/to/studies /path/to/results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Help message
show_help() {
    echo "Usage: ./run-batch.sh [INPUT_DIR] [OUTPUT_DIR]"
    echo ""
    echo "Arguments:"
    echo "  INPUT_DIR   Directory with ZIP files (default: ./input)"
    echo "  OUTPUT_DIR  Directory for results (default: ./output)"
    echo ""
    echo "Examples:"
    echo "  ./run-batch.sh                                    # Use default ./input and ./output"
    echo "  ./run-batch.sh /data/studies                      # Input from /data/studies, output to ./output"
    echo "  ./run-batch.sh /data/studies /data/results        # Custom input and output"
    echo ""
}

# Parse arguments
INPUT_DIR="${1:-$SCRIPT_DIR/input}"
OUTPUT_DIR="${2:-$SCRIPT_DIR/output}"

if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Convert to absolute paths
INPUT_DIR="$(cd "$INPUT_DIR" 2>/dev/null && pwd || echo "$INPUT_DIR")"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" 2>/dev/null && pwd || echo "$OUTPUT_DIR")"

echo -e "${GREEN}🚀 RadiAssist Batch Processing${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo -e "📁 Input:  ${YELLOW}$INPUT_DIR${NC}"
echo -e "📁 Output: ${YELLOW}$OUTPUT_DIR${NC}"
echo ""

# Check input directory exists
if [[ ! -d "$INPUT_DIR" ]]; then
    echo -e "${RED}❌ Input directory does not exist: $INPUT_DIR${NC}"
    exit 1
fi

# Create output directory if doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check for ZIP files
ZIP_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -name "*.zip" | wc -l)
if [[ $ZIP_COUNT -eq 0 ]]; then
    echo -e "${RED}❌ No ZIP files found in $INPUT_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Found $ZIP_COUNT ZIP file(s) to process${NC}"
echo ""

# Detect GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}🎮 GPU detected, using GPU acceleration${NC}"
    COMPOSE_FILE="docker-compose.batch.yml"
else
    echo -e "${YELLOW}⚠️  No GPU detected, using CPU mode${NC}"
    COMPOSE_FILE="docker-compose.batch.yml"
fi

echo ""
echo -e "${GREEN}🐳 Starting Docker container...${NC}"
echo ""

# Run docker-compose
cd "$SCRIPT_DIR"
INPUT_DIR="$INPUT_DIR" OUTPUT_DIR="$OUTPUT_DIR" docker-compose -f "$COMPOSE_FILE" up --build

echo ""
echo -e "${GREEN}✅ Batch processing completed!${NC}"
echo -e "${GREEN}📊 Results saved to: ${YELLOW}$OUTPUT_DIR${NC}"
echo ""
