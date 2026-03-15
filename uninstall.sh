#!/bin/bash

# owlTrail Uninstaller
# Entfernt Symlink, Wrapper und optional VENV

echo "🦉 owlTrail Deinstallation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 1. Server stoppen (falls läuft)
if [ -f "$BASE_DIR/.owltrail.pid" ]; then
    PID=$(cat "$BASE_DIR/.owltrail.pid")
    echo "⏹️  Stoppe Server (PID $PID)..."
    kill "$PID" 2>/dev/null
    rm -f "$BASE_DIR/.owltrail.pid"
fi

# 2. Symlink entfernen
LINK="$HOME/.local/bin/owlTrail"
if [ -L "$LINK" ]; then
    rm -f "$LINK"
    echo "✅ Symlink entfernt: $LINK"
fi

# 3. Wrapper entfernen
if [ -f "$BASE_DIR/owlTrail" ]; then
    rm -f "$BASE_DIR/owlTrail"
    echo "✅ Wrapper entfernt"
fi

# 4. Optional: VENV entfernen
echo ""
read -p "VENV auch löschen? (j/N): " ANSWER
if [ "$ANSWER" = "j" ] || [ "$ANSWER" = "J" ]; then
    rm -rf "$BASE_DIR/venv"
    echo "✅ VENV gelöscht"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ owlTrail deinstalliert."
echo "   Quellcode bleibt erhalten in: $BASE_DIR"
