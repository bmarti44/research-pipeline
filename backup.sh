#!/bin/bash

# File Backup Script
# Usage: ./backup.sh [source_directory] [backup_directory]

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to display usage
usage() {
    echo "Usage: $0 [source_directory] [backup_directory]"
    echo "Example: $0 /home/user/documents /home/user/backups"
    exit 1
}

# Check if arguments are provided
if [ $# -ne 2 ]; then
    echo -e "${RED}Error: Invalid number of arguments${NC}"
    usage
fi

SOURCE_DIR="$1"
BACKUP_DIR="$2"

# Verify source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo -e "${RED}Error: Source directory '$SOURCE_DIR' does not exist${NC}"
    exit 1
fi

# Create backup directory if it doesn't exist
if [ ! -d "$BACKUP_DIR" ]; then
    echo -e "${YELLOW}Creating backup directory: $BACKUP_DIR${NC}"
    mkdir -p "$BACKUP_DIR"
fi

# Create timestamp for backup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="backup_$TIMESTAMP.tar.gz"
BACKUP_PATH="$BACKUP_DIR/$BACKUP_NAME"

# Display backup information
echo -e "${GREEN}Starting backup...${NC}"
echo "Source: $SOURCE_DIR"
echo "Destination: $BACKUP_PATH"
echo ""

# Create the backup
tar -czf "$BACKUP_PATH" -C "$(dirname "$SOURCE_DIR")" "$(basename "$SOURCE_DIR")" 2>/dev/null

# Check if backup was successful
if [ $? -eq 0 ]; then
    BACKUP_SIZE=$(du -h "$BACKUP_PATH" | cut -f1)
    echo -e "${GREEN}✓ Backup completed successfully!${NC}"
    echo "Backup file: $BACKUP_PATH"
    echo "Size: $BACKUP_SIZE"

    # Keep only last 5 backups
    echo ""
    echo -e "${YELLOW}Cleaning old backups (keeping last 5)...${NC}"
    cd "$BACKUP_DIR"
    ls -t backup_*.tar.gz 2>/dev/null | tail -n +6 | xargs -r rm -f

    REMAINING=$(ls -1 backup_*.tar.gz 2>/dev/null | wc -l)
    echo "Total backups in directory: $REMAINING"
else
    echo -e "${RED}✗ Backup failed!${NC}"
    exit 1
fi
