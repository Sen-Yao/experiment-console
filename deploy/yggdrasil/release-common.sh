#!/usr/bin/env bash

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

env_value() {
  local file="$1" key="$2" value count
  count="$(awk -F= -v key="$key" '$1 == key {count++} END {print count+0}' "$file")"
  [[ "$count" == "1" ]] || fail "$key must appear exactly once in $file"
  value="$(awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print}' "$file")"
  [[ -n "$value" ]] || fail "$key is empty in $file"
  printf '%s' "$value"
}

require_safe_absolute_path() {
  [[ "$1" == /* ]] || fail "path must be absolute: $1"
  [[ "$1" =~ ^/[A-Za-z0-9._/-]+$ ]] || fail "path contains unsupported characters: $1"
}

load_production_context() {
  BASE="$1"
  RELEASE="$2"
  require_safe_absolute_path "$BASE"
  require_safe_absolute_path "$RELEASE"
  ENV_FILE="$BASE/config/production.env"
  COMPOSE_FILE="$RELEASE/compose.yggdrasil.yaml"
  [[ -r "$ENV_FILE" ]] || fail "production env is missing: $ENV_FILE"
  [[ -r "$COMPOSE_FILE" ]] || fail "compose file is missing: $COMPOSE_FILE"

  CONSOLE_UID="$(env_value "$ENV_FILE" EXPERIMENT_CONSOLE_UID)"
  CONSOLE_GID="$(env_value "$ENV_FILE" EXPERIMENT_CONSOLE_GID)"
  STATE_PATH="$(env_value "$ENV_FILE" EXPERIMENT_CONSOLE_STATE_PATH)"
  RESULTS_PATH="$(env_value "$ENV_FILE" EXPERIMENT_CONSOLE_RESULTS_PATH)"
  WANDB_SECRET_FILE="$(env_value "$ENV_FILE" EXPERIMENT_CONSOLE_WANDB_SECRET_FILE)"
  CONSOLE_API_TOKEN_SECRET_FILE="$(env_value "$ENV_FILE" EXPERIMENT_CONSOLE_API_TOKEN_SECRET_FILE)"
  HCCS_KEY_FILE="$(env_value "$ENV_FILE" EXPERIMENT_CONSOLE_HCCS_SSH_KEY_FILE)"
  HCCS_CONFIG_FILE="$(env_value "$ENV_FILE" EXPERIMENT_CONSOLE_HCCS_SSH_CONFIG_FILE)"
  HCCS_KNOWN_HOSTS_FILE="$(env_value "$ENV_FILE" EXPERIMENT_CONSOLE_HCCS_KNOWN_HOSTS_FILE)"
  AUTHORITY_ROLE="$(env_value "$ENV_FILE" EXPERIMENT_CONSOLE_AUTHORITY_ROLE)"
  INSTANCE_ID="$(env_value "$ENV_FILE" EXPERIMENT_CONSOLE_INSTANCE_ID)"

  [[ "$CONSOLE_UID" =~ ^[0-9]+$ ]] || fail "EXPERIMENT_CONSOLE_UID must be numeric"
  [[ "$CONSOLE_GID" =~ ^[0-9]+$ ]] || fail "EXPERIMENT_CONSOLE_GID must be numeric"
  [[ "$AUTHORITY_ROLE" == "authoritative" ]] || fail "production authority role must be authoritative"
  [[ "$INSTANCE_ID" != "local-development" ]] || fail "production instance id must not identify a development Console"
  for path in "$STATE_PATH" "$RESULTS_PATH" "$WANDB_SECRET_FILE" "$CONSOLE_API_TOKEN_SECRET_FILE" "$HCCS_KEY_FILE" "$HCCS_CONFIG_FILE" "$HCCS_KNOWN_HOSTS_FILE"; do
    require_safe_absolute_path "$path"
  done

  RELEASE_NAME="$(basename "$RELEASE")"
  [[ "$RELEASE_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || fail "release name cannot form a Docker tag: $RELEASE_NAME"
  IMAGE_TAG="release-$RELEASE_NAME"
  [[ "${#IMAGE_TAG}" -le 128 ]] || fail "release image tag is too long"
  COMPOSE=(env "EXPERIMENT_CONSOLE_IMAGE_TAG=$IMAGE_TAG" docker compose --project-name experiment-console --project-directory "$RELEASE" --env-file "$ENV_FILE" -f "$COMPOSE_FILE")
}

current_release() {
  if [[ -L "$1/current" ]]; then
    readlink -f "$1/current"
  fi
}

container_id() {
  "${COMPOSE[@]}" ps --quiet console
}
