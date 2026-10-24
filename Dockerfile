FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
# Artifacts created at runtime if missing (train-on-boot)
ENV HOST=0.0.0.0 PORT=8091 MODEL_DIR=/app/artifacts
EXPOSE 8091
CMD ["model-ops-serve"]
