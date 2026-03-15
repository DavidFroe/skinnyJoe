#!/bin/bash
# ================================================================
# SkinnyJoe Git Download — Pull von GitHub
# ================================================================

set -e
cd "$(dirname "$(readlink -f "$0")")"

echo "🛬 SkinnyJoe Git Download"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

BRANCH="main"

echo "⚠️  Alle lokalen Änderungen werden überschrieben!"
echo -n "Fortfahren? [j/N]: "
read -r CONFIRM
if [[ ! "$CONFIRM" =~ ^[jJyY]$ ]]; then
    echo "Abgebrochen."
    exit 0
fi

echo "📥 Hole Stand von origin/$BRANCH..."
git fetch --all
git checkout main 2>/dev/null || true
git reset --hard "origin/$BRANCH"
git clean -fd

chmod +x *.sh 2>/dev/null || true

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Download von 'main' erledigt!"
echo "   Tipp: sj server restart"
