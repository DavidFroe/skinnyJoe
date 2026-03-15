#!/bin/bash
# SkinnyJoe Installer
#
# Nutzung:
#   ./install.sh                  Normaler Installations-Wizard
#   ./install.sh --autostart-on   Autostart aktivieren (headless)
#   ./install.sh --autostart-off  Autostart deaktivieren (headless)
#   ./install.sh --autostart-status  Autostart-Status anzeigen

set -euo pipefail

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$BASE_DIR"

# ---------------------------------------------------------------------------
# Autostart-Funktionen (systemd user service, crontab fallback)
# ---------------------------------------------------------------------------

_SVC_NAME="skinnyjoe"
_SVC_DIR="$HOME/.config/systemd/user"
_SVC_FILE="$_SVC_DIR/${_SVC_NAME}.service"

_detect_init() {
    if command -v systemctl &>/dev/null && systemctl --user status &>/dev/null 2>&1; then
        echo "systemd"
    else
        echo "crontab"
    fi
}

_autostart_on() {
    local init_sys venv_py daemon_script
    init_sys=$(_detect_init)
    venv_py="$BASE_DIR/venv/bin/python3"
    daemon_script="$BASE_DIR/skinnyJoe_daemon.py"

    if [ ! -f "$venv_py" ] || [ ! -f "$daemon_script" ]; then
        echo "❌ SkinnyJoe ist nicht vollständig installiert."
        echo "   Bitte zuerst ./install.sh ohne Parameter ausführen."
        exit 1
    fi

    if [ "$init_sys" = "systemd" ]; then
        echo "📦 Erstelle systemd User-Service..."
        mkdir -p "$_SVC_DIR"
        cat > "$_SVC_FILE" << SVCEOF
[Unit]
Description=SkinnyJoe Multi-Slot KI-Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$BASE_DIR
ExecStart=$venv_py $daemon_script
Restart=on-failure
RestartSec=5
Environment=HOME=$HOME

[Install]
WantedBy=default.target
SVCEOF
        systemctl --user daemon-reload
        systemctl --user enable "$_SVC_NAME"
        echo "✅ Autostart aktiviert (systemd user service)"
        echo "   Service: $_SVC_FILE"
        echo ""
        echo "   Befehle:"
        echo "   - Status:  systemctl --user status $_SVC_NAME"
        echo "   - Log:     journalctl --user -u $_SVC_NAME -f"
        echo "   - Stopp:   systemctl --user stop $_SVC_NAME"
        if command -v loginctl &>/dev/null; then
            loginctl enable-linger "$(whoami)" 2>/dev/null && \
                echo "   - Linger:  aktiviert (Service läuft auch ohne Login)" || true
        fi
    else
        echo "📦 Kein systemd verfügbar, nutze crontab..."
        local cron_cmd="@reboot cd $BASE_DIR && $venv_py $daemon_script >> $BASE_DIR/skinnyjoe.log 2>&1"
        if crontab -l 2>/dev/null | grep -qF "skinnyJoe_daemon.py"; then
            echo "⚠️  Autostart ist bereits in crontab eingetragen."
            return
        fi
        (crontab -l 2>/dev/null; echo "$cron_cmd") | crontab -
        echo "✅ Autostart aktiviert (crontab @reboot)"
    fi
}

_autostart_off() {
    local init_sys removed
    init_sys=$(_detect_init)
    removed=false

    if [ "$init_sys" = "systemd" ] && [ -f "$_SVC_FILE" ]; then
        echo "🔧 Deaktiviere systemd User-Service..."
        systemctl --user stop "$_SVC_NAME" 2>/dev/null || true
        systemctl --user disable "$_SVC_NAME" 2>/dev/null || true
        rm -f "$_SVC_FILE"
        systemctl --user daemon-reload
        echo "✅ Autostart deaktiviert (systemd service entfernt)"
        removed=true
    fi

    if crontab -l 2>/dev/null | grep -qF "skinnyJoe_daemon.py"; then
        echo "🔧 Entferne crontab-Eintrag..."
        crontab -l 2>/dev/null | grep -vF "skinnyJoe_daemon.py" | crontab -
        echo "✅ Autostart deaktiviert (crontab-Eintrag entfernt)"
        removed=true
    fi

    if [ "$removed" = false ]; then
        echo "ℹ️  Kein Autostart konfiguriert — nichts zu tun."
    fi
}

_autostart_status() {
    local init_sys active
    init_sys=$(_detect_init)
    active=false

    if [ "$init_sys" = "systemd" ] && [ -f "$_SVC_FILE" ]; then
        if systemctl --user is-enabled "$_SVC_NAME" &>/dev/null; then
            echo "✅ Autostart aktiv (systemd user service)"
            systemctl --user status "$_SVC_NAME" --no-pager 2>/dev/null || true
            active=true
        fi
    fi

    if crontab -l 2>/dev/null | grep -qF "skinnyJoe_daemon.py"; then
        echo "✅ Autostart aktiv (crontab @reboot)"
        crontab -l 2>/dev/null | grep -F "skinnyJoe_daemon.py"
        active=true
    fi

    if [ "$active" = false ]; then
        echo "❌ Autostart nicht konfiguriert"
        echo "   Aktivieren: sj server autostart on"
    fi
}

# ---------------------------------------------------------------------------
# Headless-Modus: nur Autostart-Verwaltung
# ---------------------------------------------------------------------------
case "${1:-}" in
    --autostart-on)
        echo "SkinnyJoe Autostart"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        _autostart_on
        exit 0
        ;;
    --autostart-off)
        echo "SkinnyJoe Autostart"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        _autostart_off
        exit 0
        ;;
    --autostart-status)
        echo "SkinnyJoe Autostart-Status"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        _autostart_status
        exit 0
        ;;
esac

# ---------------------------------------------------------------------------
# Normaler Installations-Wizard
# ---------------------------------------------------------------------------

echo "SkinnyJoe Installation (intelligenter Modus)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
START_TIME=$(date +%s)

LOG_FILE="$BASE_DIR/install_debug.log"
echo "Alle Logs: $LOG_FILE"
echo "========== LOG START $(date) ==========" > "$LOG_FILE"

log() {
    echo -e "$@" | tee -a "$LOG_FILE"
}

section() {
    echo "" | tee -a "$LOG_FILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
    log "$1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG_FILE"
}

section "Schritt 1: Python & venv prüfen"

if ! command -v python3 >/dev/null 2>&1; then
    log "FEHLER: python3 nicht gefunden. Bitte Python 3 installieren."
    exit 1
fi

PY_VER=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:3])))")
log "Verwendetes Python: $PY_VER"

# 1. Python VENV
if [ ! -d "venv" ]; then
    log "Erstelle Virtual Environment in $BASE_DIR/venv ..."
    python3 -m venv venv 2>&1 | tee -a "$LOG_FILE"
else
    log "VENV existiert bereits – verwende bestehendes venv."
fi

VENV_PY="$BASE_DIR/venv/bin/python3"
VENV_PIP="$BASE_DIR/venv/bin/pip"

if [ ! -x "$VENV_PY" ]; then
    log "FEHLER: venv scheint defekt zu sein (python3 nicht ausführbar)."
    log "Lösche venv und starte neu: rm -rf venv && ./install.sh"
    exit 1
fi

section "Schritt 2: pip aktualisieren"

"$VENV_PY" -m pip install --upgrade pip wheel setuptools 2>&1 | tee -a "$LOG_FILE"
log "pip Version in venv:"
"$VENV_PIP" --version | tee -a "$LOG_FILE"

section "Schritt 3: Basis-Abhängigkeiten installieren"

BASE_PACKAGES=(
    "fastapi"
    "uvicorn"
    "pydantic"
    "llama-cpp-python"
    "requests"
)

FAILED_BASE=()
for PKG in "${BASE_PACKAGES[@]}"; do
    log "==> Installiere Basis-Paket: $PKG"
    if ! "$VENV_PIP" install "$PKG" 2>&1 | tee -a "$LOG_FILE"; then
        log "!! FEHLER beim Installieren von Basis-Paket: $PKG"
        FAILED_BASE+=("$PKG")
    fi
done

if [ ${#FAILED_BASE[@]} -gt 0 ]; then
    log "WARNUNG: Einige Basis-Pakete konnten nicht installiert werden:"
    for PKG in "${FAILED_BASE[@]}"; do
        log "  - $PKG"
    done
else
    log "Alle Basis-Pakete wurden erfolgreich installiert."
fi

section "Schritt 4: Vollständiges requirements.txt versuchen"

if [ -f "requirements.txt" ]; then
    log "Versuche vollständige Installation aus requirements.txt (mit Debug-Ausgabe)..."
    if ! "$VENV_PIP" install -r requirements.txt 2>&1 | tee -a "$LOG_FILE"; then
        log "!! FEHLER: pip install -r requirements.txt ist gescheitert."
        log "Wir versuchen nun, kritische Heavy-ML-Pakete einzeln zu installieren."
    else
        log "requirements.txt wurde vollständig installiert."
    fi
else
    log "WARNUNG: requirements.txt nicht gefunden – überspringe diesen Schritt."
fi

section "Schritt 5: Heavy-ML-Pakete einzeln testen"

HEAVY_PACKAGES=(
    "torch"
    "diffusers"
    "transformers"
    "accelerate"
    "safetensors"
    "sentencepiece"
    "Pillow"
)

FAILED_HEAVY=()
for PKG in "${HEAVY_PACKAGES[@]}"; do
    log "==> Prüfe/Installiere Heavy-Paket: $PKG"
    if "$VENV_PY" -c "import importlib; exit(0) if importlib.util.find_spec('$PKG'.split('[')[0]) else exit(1)" 2>/dev/null; then
        log "   $PKG ist bereits installiert (import erfolgreich)."
        continue
    fi

    if [ "$PKG" = "torch" ]; then
        log "   Spezielle Installation für torch (CPU-Index)..."
        if ! "$VENV_PIP" install torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tee -a "$LOG_FILE"; then
            log "!! FEHLER bei Installation von torch (CPU)."
            FAILED_HEAVY+=("$PKG")
        fi
    else
        if ! "$VENV_PIP" install "$PKG" 2>&1 | tee -a "$LOG_FILE"; then
            log "!! FEHLER bei Installation von Heavy-Paket: $PKG"
            FAILED_HEAVY+=("$PKG")
        fi
    fi
done

section "Schritt 6: Skripte ausführbar machen"

chmod +x skinnyJoe_cli.py skinnyJoe_daemon.py skinnyJoe_tui.py 2>/dev/null || true

section "Schritt 7: Wrapper-Skripte erzeugen"

cat > "$BASE_DIR/sj" << 'WRAPPER'
#!/bin/bash
BASE_DIR="$( cd "$( dirname "$(readlink -f "${BASH_SOURCE[0]}")" )" &> /dev/null && pwd )"
exec "$BASE_DIR/venv/bin/python3" "$BASE_DIR/skinnyJoe_cli.py" "$@"
WRAPPER
chmod +x "$BASE_DIR/sj"

cat > "$BASE_DIR/sj-daemon" << 'WRAPPER'
#!/bin/bash
BASE_DIR="$( cd "$( dirname "$(readlink -f "${BASH_SOURCE[0]}")" )" &> /dev/null && pwd )"
exec "$BASE_DIR/venv/bin/python3" "$BASE_DIR/skinnyJoe_daemon.py" "$@"
WRAPPER
chmod +x "$BASE_DIR/sj-daemon"

section "Schritt 8: Symlinks in ~/.local/bin"

mkdir -p "$HOME/.local/bin"
for CMD in sj sj-daemon; do
    LINK_TARGET="$HOME/.local/bin/$CMD"
    rm -f "$LINK_TARGET"
    ln -s "$BASE_DIR/$CMD" "$LINK_TARGET"
    log "Symlink erstellt: $LINK_TARGET -> $BASE_DIR/$CMD"
done

if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo ""
    log "HINWEIS: $HOME/.local/bin ist nicht im PATH!"
    log "   Füge diese Zeile in ~/.bashrc ein:"
    log "   export PATH=\"$HOME/.local/bin:\$PATH\""
    echo ""
fi

section "Zusammenfassung"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ ${#FAILED_BASE[@]} -eq 0 ] && [ ${#FAILED_HEAVY[@]} -eq 0 ]; then
    log "Installation abgeschlossen – alle Pakete wurden (soweit bekannt) erfolgreich installiert."
else
    log "Installation abgeschlossen, aber einige Pakete sind fehlgeschlagen:"
    if [ ${#FAILED_BASE[@]} -gt 0 ]; then
        log "  Basis-Pakete fehlgeschlagen:"
        for PKG in "${FAILED_BASE[@]}"; do
            log "    - $PKG"
        done
    fi
    if [ ${#FAILED_HEAVY[@]} -gt 0 ]; then
        log "  Heavy-ML-Pakete fehlgeschlagen:"
        for PKG in "${FAILED_HEAVY[@]}"; do
            log "    - $PKG"
        done
    fi
    log "Siehe Detail-Log: $LOG_FILE"
fi

echo ""
log "Installationsdauer: ${DURATION}s"
echo ""
echo "Befehle:"
echo "  sj server start                 Daemon starten"
echo "  sj server stop                  Daemon stoppen"
echo "  sj server status                Status + Slots anzeigen"
echo "  sj server log                   Live-Log anzeigen"
echo "  sj server autostart on          Autostart beim Boot aktivieren"
echo "  sj server autostart off         Autostart deaktivieren"
echo "  sj models                       Modelle + Profile anzeigen"
echo "  sj slots                        Slot-Status anzeigen"
echo "  sj gpus                         NVIDIA GPUs anzeigen"
echo "  sj load 5 --slot 1              Modell N5 in Slot 1 laden"
echo "  sj unload --slot 1              Slot 1 entladen"
echo "  sj ask --slot 1 \"Prompt\"        Anfrage an Slot 1"
echo "  sj ask --slot 2 --image f.jpg \"Beschreibe\"  Vision"
echo "  sj tui                          Interaktive TUI"
echo "  sj status                       Gesamtstatus"
echo "  sj help                         Vollstaendige Hilfe"
echo ""

# Autostart-Wizard
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Autostart beim Systemstart einrichten?"
echo -n "  [j/N]: "
read -r AS_CONFIRM
if [[ "$AS_CONFIRM" =~ ^[jJyY]$ ]]; then
    _autostart_on
else
    echo "  Autostart nicht aktiviert."
    echo "  Später jederzeit: sj server autostart on"
fi

echo ""
echo "Schnittstellenbeschreibung: interface.out"
echo ""
echo "Detail-Log: $LOG_FILE"
echo "========== LOG ENDE $(date) ==========" >> "$LOG_FILE"
