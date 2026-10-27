FROM python:3.12-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# The trained bundle (model, encoders, velocity state, drift reference) from `make train`.
ARG ARTIFACTS=artifacts/synthetic
COPY ${ARTIFACTS} /app/model
ENV FRAUD_ARTIFACTS=/app/model

RUN useradd --create-home app
USER app
EXPOSE 8000
CMD ["uvicorn", "fraud_detection.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
