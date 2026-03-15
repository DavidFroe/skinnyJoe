#!/bin/bash
# ================================================================
# SkinnyJoe Git Upload — Push lokalen Stand zu GitHub
# ================================================================

set -e
cd "$(dirname "$(readlink -f "$0")")"

echo "🚀 SkinnyJoe Git Upload"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

BRANCH="main"
git checkout main 2>/dev/null || true

chmod +x *.sh 2>/dev/null || true

git add -A

CHANGES=$(git diff --cached --stat)
if [ -z "$CHANGES" ]; then
    echo "ℹ️  Keine Änderungen zum Hochladen."
    exit 0
fi

echo "📋 Änderungen:"
echo "$CHANGES"
echo ""

MSG="SkinnyJoe Sync: $(date +'%Y-%m-%d %H:%M:%S')"
git commit -m "$MSG"
git push origin "$BRANCH" --force

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Upload auf 'main' erledigt!"
