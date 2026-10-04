#!/bin/bash
# Open the workbook in Microsoft Excel (macOS), force a full recalculation, save and close,
# so that cached values exist for verification (python verify_excel.py).
# Fails if Excel did not actually save the file, for example when a dialog is waiting in Excel.
set -euo pipefail
FILE="${1:-../excel/Miniproject3_CDO_Analysis.xlsx}"
ABS="$(cd "$(dirname "$FILE")" && pwd)/$(basename "$FILE")"
NAME="$(basename "$FILE")"
BEFORE="$(stat -f %m "$ABS")"
sleep 1
osascript <<APPLESCRIPT
with timeout of 300 seconds
  tell application "Microsoft Excel"
    set display alerts to false
    open POSIX file "$ABS"
    set wb to workbook "$NAME"
    calculate full
    save wb
    close wb saving no
  end tell
end timeout
APPLESCRIPT
AFTER="$(stat -f %m "$ABS")"
if [ "$AFTER" = "$BEFORE" ]; then
  echo "Excel did not save $ABS. Check for a dialog waiting in Excel, then rerun." >&2
  exit 1
fi
echo "Recalculated and saved: $ABS"
