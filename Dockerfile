ARG BUILD_FROM=python:3.12-slim
FROM ${BUILD_FROM}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install uv --no-cache-dir

COPY pyproject.toml uv.lock* ./
# CPU-only torch: override the PyPI torch entry with the cpu wheel index
RUN uv sync --no-dev \
      --index "pytorch-cpu=https://download.pytorch.org/whl/cpu" \
      --override "torch=torch[cpu]"

COPY src/ src/

CMD ["uv", "run", "python", "-m", "espresso_rl.main"]
