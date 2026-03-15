#!/bin/bash
echo "🛬 Starte Download (Lokaler Ordner wird plattgemacht)..."
BRANCH=$(git branch --show-current)
git fetch --all
git reset --hard "origin/$BRANCH"
git clean -fd
echo "✅ Download erledigt!"
