#!/usr/bin/env sh
set -eu

WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-/run/secrets/wandb_api_key}"
EXPERIMENT_CONSOLE_API_TOKEN_FILE="${EXPERIMENT_CONSOLE_API_TOKEN_FILE:-/run/secrets/console_api_token}"
HCCS_SSH_KEY_FILE="${HCCS_SSH_KEY_FILE:-/run/secrets/hccs_ssh_key}"
HCCS_SSH_CONFIG_FILE="${HCCS_SSH_CONFIG_FILE:-/run/config/hccs_ssh_config}"
HCCS_KNOWN_HOSTS_FILE="${HCCS_KNOWN_HOSTS_FILE:-/run/config/hccs_known_hosts}"
SSH_DIR="${HOME:-/home/console}/.ssh"

require_readable_nonempty() {
    if [ ! -r "$1" ] || [ ! -s "$1" ]; then
        echo "Required credential/config file is missing or empty: $1" >&2
        exit 78
    fi
}

require_readable_nonempty "$WANDB_API_KEY_FILE"
require_readable_nonempty "$EXPERIMENT_CONSOLE_API_TOKEN_FILE"
require_readable_nonempty "$HCCS_SSH_KEY_FILE"
require_readable_nonempty "$HCCS_SSH_CONFIG_FILE"
require_readable_nonempty "$HCCS_KNOWN_HOSTS_FILE"
API_TOKEN_LENGTH="$(tr -d '\r\n' < "$EXPERIMENT_CONSOLE_API_TOKEN_FILE" | wc -c | tr -d ' ')"
if [ "$API_TOKEN_LENGTH" -lt 32 ]; then
    echo "Console API token must contain at least 32 non-newline characters" >&2
    exit 78
fi

umask 077
mkdir -p "$SSH_DIR"
cp "$HCCS_SSH_KEY_FILE" "$SSH_DIR/id_hccs"
cp "$HCCS_SSH_CONFIG_FILE" "$SSH_DIR/config"
cp "$HCCS_KNOWN_HOSTS_FILE" "$SSH_DIR/known_hosts"
chmod 0600 "$SSH_DIR/id_hccs" "$SSH_DIR/config" "$SSH_DIR/known_hosts"

exec "$@"
