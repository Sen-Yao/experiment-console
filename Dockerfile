ARG BASE_REGISTRY=docker.io/library

FROM ${BASE_REGISTRY}/python:3.13-slim-bookworm

ARG CONSOLE_UID=10001
ARG CONSOLE_GID=10001

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend:/app:/app/scripts \
    EXPERIMENT_CONSOLE_STATE_DIR=/var/lib/experiment-console-v3 \
    EXPERIMENT_CONSOLE_SERVER_PROFILES=/etc/experiment-console/server-profiles.json \
    EXPERIMENT_CONSOLE_INSTANCE_ID=yggdrasil-production-v3 \
    EXPERIMENT_CONSOLE_REQUIRE_API_TOKEN=1 \
    HOME=/home/console

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "$CONSOLE_GID" console \
    && useradd --uid "$CONSOLE_UID" --gid "$CONSOLE_GID" --create-home --shell /usr/sbin/nologin console \
    && install -d -o "$CONSOLE_UID" -g "$CONSOLE_GID" -m 0700 /home/console/.ssh \
    && install -d -o "$CONSOLE_UID" -g "$CONSOLE_GID" -m 0750 /var/lib/experiment-console-v3 \
    && install -d -o "$CONSOLE_UID" -g "$CONSOLE_GID" -m 0750 /etc/experiment-console

WORKDIR /app
COPY pyproject.toml ./
COPY backend/ ./backend/
RUN python -m pip install --no-cache-dir .
COPY config/server-profiles.json /etc/experiment-console/server-profiles.json
COPY scripts/runtime_console_server.py ./scripts/runtime_console_server.py
COPY deploy/yggdrasil/container-entrypoint.sh /usr/local/bin/experiment-console-entrypoint
COPY deploy/yggdrasil/container-healthcheck.py /usr/local/bin/experiment-console-healthcheck

RUN chmod 0555 /usr/local/bin/experiment-console-entrypoint /usr/local/bin/experiment-console-healthcheck \
    && chown -R console:console /app /etc/experiment-console /var/lib/experiment-console-v3

USER console:console
EXPOSE 5174
ENTRYPOINT ["/usr/local/bin/experiment-console-entrypoint"]
CMD ["python", "-m", "uvicorn", "runtime_console_server:create_app", "--factory", "--host", "0.0.0.0", "--port", "5174", "--no-server-header"]
