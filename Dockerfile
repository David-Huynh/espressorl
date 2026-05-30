ARG BUILD_FROM=python:3.12-slim
ARG BUILD_VERSION=0.1.0
ARG BUILD_ARCH=amd64
FROM ${BUILD_FROM}
ARG BUILD_VERSION=0.1.0
ARG BUILD_ARCH=amd64

LABEL \
    io.hass.version="${BUILD_VERSION}" \
    io.hass.type="app" \
    io.hass.arch="${BUILD_ARCH}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install uv --no-cache-dir

COPY pyproject.toml uv.lock* README.md ./
COPY src/ src/
# CPU-only torch is selected by the [tool.uv.sources] entry in pyproject.toml.
RUN uv sync --no-dev --frozen

CMD ["uv", "run", "python", "-m", "espresso_rl.main"]
