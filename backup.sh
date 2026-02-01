#!/bin/bash

# File Backup Script
# Usage: ./backup.sh <source_directory> <backup_destination>

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to display usage
usage() {
    echo "Usage: $0 <source_directory> <backup_destination>"
    echo "Example: $0 /home/user/documents /home/user/backups"
    exit 1
}

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo -e "${RED}Error: Invalid number of arguments${NC}"
    usage
fi

SOURCE_DIR="$1"
BACKUP_DEST="$2"

# Validate source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo -e "${RED}Error: Source directory '$SOURCE_DIR' does not exist${NC}"
    exit 1
fi

# Create backup destination if it doesn't exist
if [ ! -d "$BACKUP_DEST" ]; then
    echo -e "${YELLOW}Backup destination does not exist. Creating it...${NC}"
    mkdir -p "$BACKUP_DEST"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to create backup destination${NC}"
        exit 1
    fi
fi

# Generate timestamp for backup folder
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="backup_${TIMESTAMP}"
BACKUP_PATH="${BACKUP_DEST}/${BACKUP_NAME}"

# Create backup
echo -e "${GREEN}Starting backup...${NC}"
echo "Source: $SOURCE_DIR"
echo "Destination: $BACKUP_PATH"
echo ""

# Create the backup using tar with compression
TAR_FILE="${BACKUP_PATH}.tar.gz"
tar -czf "$TAR_FILE" -C "$(dirname "$SOURCE_DIR")" "$(basename "$SOURCE_DIR")" 2>/dev/null

if [ $? -eq 0 ]; then
    BACKUP_SIZE=$(du -h "$TAR_FILE" | cut -f1)
    echo -e "${GREEN}✓ Backup completed successfully!${NC}"
    echo "Backup file: $TAR_FILE"
    echo "Backup size: $BACKUP_SIZE"

    # Create a symlink to the latest backup
    LATEST_LINK="${BACKUP_DEST}/latest_backup.tar.gz"
    ln -sf "$TAR_FILE" "$LATEST_LINK"
    echo "Latest backup link: $LATEST_LINK"

    # Optional: Keep only last N backups (uncomment to enable)
    # MAX_BACKUPS=5
    # cd "$BACKUP_DEST"
    # ls -t backup_*.tar.gz | tail -n +$((MAX_BACKUPS + 1)) | xargs -r rm
    # echo "Cleaned up old backups (keeping last $MAX_BACKUPS)"

else
    echo -e "${RED}✗ Backup failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Backup process completed!${NC}"
