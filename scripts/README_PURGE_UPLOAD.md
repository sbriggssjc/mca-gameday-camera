# Purge and Upload Script

`purge_and_upload.sh` analyzes recording storage, optionally archives old segments, uploads to Google Drive via [`rclone`](https://rclone.org), and purges local files.

## Prerequisites

```bash
sudo apt-get update && sudo apt-get install -y rclone
rclone config   # create a remote, e.g. gdrive:
```

## Examples

Keep the newest 16 files per extension and upload results:

```bash
./scripts/purge_and_upload.sh --keep 16 --upload --remote gdrive: --dest "mca-gameday-camera/backups"
```

Purge files older than 7 days (dry-run):

```bash
./scripts/purge_and_upload.sh --days 7 --dry-run
```

Cap directory at 10 GB:

```bash
./scripts/purge_and_upload.sh --max-size 10
```

Example cron entry (daily at 2:15am):

```
15 2 * * * /home/scott/mca-gameday-camera/scripts/purge_and_upload.sh --keep 16 --upload --remote gdrive: --dest "mca-gameday-camera/backups" >> /home/scott/mca-gameday-camera/logs/maintenance/cron.log 2>&1
```

