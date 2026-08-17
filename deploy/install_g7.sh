#!/usr/bin/env bash
# Installs the digest on host-b: a venv, a daily systemd timer, and the
# nginx container the Cloudflare Tunnel points at. Idempotent, safe to re-run.
#
# The tunnel ingress rule and the DNS record are deliberately NOT created here.
# They are edge changes with a blast radius beyond this app, and the runbook in
# README.md walks them once.
set -euo pipefail

APP_DIR=${APP_DIR:-/home/deploy/arxiv-digest}
BIND_IP=${BIND_IP:-100.100.100.100}
PORT=${PORT:-4245}
# Ten is the cap in cli.py, set by the 100,000 token daily budget. Selection
# costs ~6,000 tokens and each paper ~4,900, so ten lands near 55,000 and
# leaves room for the check-failure retries that re-send a whole paper.
COUNT=${COUNT:-10}
RUN_AT=${RUN_AT:-07:00}
TZ_NAME=${TZ_NAME:-America/Halifax}
CONTAINER=${CONTAINER:-arxiv-digest-web}

cd "$APP_DIR"

echo "==> python environment"
python3 -m venv .venv
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -e .

mkdir -p "$APP_DIR/site" "$APP_DIR/digests/data"

# An empty site directory makes nginx answer 403 on the root, which reads like
# a permissions fault rather than "no digests yet". Render whatever is archived,
# and an empty index if nothing is.
./.venv/bin/python -m arxiv_digest --rebuild-site \
    --out-dir "$APP_DIR/digests" --site-dir "$APP_DIR/site" 2>/dev/null \
    || ./.venv/bin/python -c "from pathlib import Path; from arxiv_digest import site; site.build([], Path('$APP_DIR/site'))"

if [ ! -f "$APP_DIR/.env" ]; then
    cat > "$APP_DIR/.env" <<'ENVEOF'
# Free hosted tier. Get a key at https://console.groq.com/keys
ARXIV_DIGEST_BACKEND=openai
ARXIV_DIGEST_API_KEY=
ARXIV_DIGEST_MODEL=openai/gpt-oss-120b
ARXIV_DIGEST_BASE_URL=https://api.groq.com/openai/v1
ENVEOF
    chmod 600 "$APP_DIR/.env"
    echo "    wrote .env, the key still has to go in it"
fi

echo "==> systemd unit and timer"
sudo tee /etc/systemd/system/arxiv-digest.service >/dev/null <<EOF
[Unit]
Description=Daily arXiv AI digest
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=deploy
WorkingDirectory=$APP_DIR
# The box runs on UTC. Without this the digest published on a Halifax evening
# is dated tomorrow, because date.today() answers in the server's zone.
Environment=TZ=$TZ_NAME
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/.venv/bin/python -m arxiv_digest -n $COUNT --out-dir $APP_DIR/digests --site-dir $APP_DIR/site
EOF

sudo tee /etc/systemd/system/arxiv-digest.timer >/dev/null <<EOF
[Unit]
Description=Run the arXiv digest every morning

[Timer]
# The timezone is spelled out because g7 runs on UTC. Without it, 07:00 here
# means 04:00 in Halifax.
OnCalendar=*-*-* $RUN_AT:00 $TZ_NAME
# No RandomizedDelaySec. A spread start is for fleets that would stampede a
# shared service; this is one box making one arXiv request, so the spread only
# bought an unpredictable publish time somewhere in a 15 minute window. The run
# takes well under a minute, so a fixed start puts the page up by $RUN_AT plus one.
Persistent=true

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now arxiv-digest.timer

echo "==> web container"
# Bound to the tailnet address only. Cloudflare Tunnel is the sole public path,
# which is the same containment every other site on this box uses.
sudo docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
sudo docker run -d \
    --name "$CONTAINER" \
    --restart unless-stopped \
    --read-only \
    --tmpfs /var/cache/nginx --tmpfs /var/run --tmpfs /tmp \
    --memory 64m \
    -v "$APP_DIR/site:/usr/share/nginx/html:ro" \
    -p "$BIND_IP:$PORT:80" \
    nginx:alpine >/dev/null

echo
echo "Installed."
echo "  site      $APP_DIR/site  ->  http://$BIND_IP:$PORT"
echo "  timer     $(systemctl show -p NextElapseUSecRealtime --value arxiv-digest.timer)"
echo "  run now   sudo systemctl start arxiv-digest.service"
echo "  logs      journalctl -u arxiv-digest.service -n 50"
