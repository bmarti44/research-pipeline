#!/bin/bash

#############################################
# File Backup Script
# Usage: ./backup.sh [source_directory] [backup_directory]
#############################################

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to display usage
usage() {
    echo "Usage: $0 [source_directory] [backup_directory]"
    echo ""
    echo "Options:"
    echo "  source_directory   - Directory to backup (default: current directory)"
    echo "  backup_directory   - Directory where backup will be stored (default: ./backups)"
    echo ""
    echo "Example:"
    echo "  $0 /home/user/documents /home/user/backups"
    exit 1
}

# Function to log messages
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Parse arguments
SOURCE_DIR="${1:-.}"
BACKUP_DIR="${2:-./backups}"

# Validate source directory
if [ ! -d "$SOURCE_DIR" ]; then
    error "Source directory '$SOURCE_DIR' does not exist!"
    exit 1
fi

# Create backup directory if it doesn't exist
if [ ! -d "$BACKUP_DIR" ]; then
    log "Creating backup directory: $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
fi

# Generate timestamp for backup file
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
SOURCE_BASENAME=$(basename "$SOURCE_DIR")
BACKUP_FILE="${BACKUP_DIR}/${SOURCE_BASENAME}_backup_${TIMESTAMP}.tar.gz"

# Start backup
log "Starting backup of '$SOURCE_DIR'..."
log "Backup destination: $BACKUP_FILE"

# Create compressed archive
if tar -czf "$BACKUP_FILE" -C "$(dirname "$SOURCE_DIR")" "$(basename "$SOURCE_DIR")" 2>/dev/null; then
    BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    log "Backup completed successfully!"
    log "Backup size: $BACKUP_SIZE"
    log "Backup location: $BACKUP_FILE"

    # Display backup file info
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Backup Summary:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Source:      $SOURCE_DIR"
    echo "Destination: $BACKUP_FILE"
    echo "Size:        $BACKUP_SIZE"
    echo "Timestamp:   $(date +'%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Optional: Keep only last N backups (uncomment to enable)
    # MAX_BACKUPS=5
    # log "Cleaning up old backups (keeping last $MAX_BACKUPS)..."
    # ls -t "${BACKUP_DIR}/${SOURCE_BASENAME}_backup_"*.tar.gz 2>/dev/null | tail -n +$((MAX_BACKUPS + 1)) | xargs -r rm

    exit 0
else
    error "Backup failed!"
    exit 1
fi
