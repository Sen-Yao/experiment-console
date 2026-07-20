#!/usr/bin/env sh
set -eu

TOKEN_FILE="${EXPERIMENT_CONSOLE_API_TOKEN_FILE:-/run/secrets/console_api_token}"
SSH_KEY_FILE="${EXPERIMENT_CONSOLE_SSH_KEY_FILE:-/run/secrets/hccs_ssh_key}"
SSH_CONFIG_FILE="${EXPERIMENT_CONSOLE_SSH_CONFIG_FILE:-/run/config/ssh_config}"
KNOWN_HOSTS_FILE="${EXPERIMENT_CONSOLE_KNOWN_HOSTS_FILE:-/run/config/known_hosts}"
SSH_DIR="${HOME:-/home/console}/.ssh"

for path in "$TOKEN_FILE" "$SSH_KEY_FILE" "$SSH_CONFIG_FILE" "$KNOWN_HOSTS_FILE"; do
    if [ ! -s "$path" ]; then
        echo "Required v3 secret/config is missing: $path" >&2
        exit 78
    fi
done

if [ "$(wc -c < "$TOKEN_FILE" | tr -d ' ')" -lt 32 ]; then
    echo "Console API token must contain at least 32 bytes" >&2
    exit 78
fi

umask 077
mkdir -p "$SSH_DIR"
cp "$SSH_KEY_FILE" "$SSH_DIR/id_console"
cp "$SSH_CONFIG_FILE" "$SSH_DIR/config"
cp "$KNOWN_HOSTS_FILE" "$SSH_DIR/known_hosts"
chmod 0600 "$SSH_DIR/id_console" "$SSH_DIR/config" "$SSH_DIR/known_hosts"
export EXPERIMENT_CONSOLE_SSH_KEY_FILE="$SSH_DIR/id_console"
export EXPERIMENT_CONSOLE_SSH_CONFIG_FILE="$SSH_DIR/config"
export EXPERIMENT_CONSOLE_KNOWN_HOSTS_FILE="$SSH_DIR/known_hosts"
exec "$@"
