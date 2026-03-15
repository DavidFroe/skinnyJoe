#!/bin/bash
echo "🚀 Starte Upload (Server wird plattgemacht)..."
BRANCH=$(git branch --show-current)
git add -A
git commit -m "Auto-Sync: $(date +'%Y-%m-%d %H:%M:%S')"
git push origin "$BRANCH" --force
echo "✅ Upload erledigt!"
