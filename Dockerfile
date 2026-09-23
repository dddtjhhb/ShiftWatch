FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY shiftwatch ./shiftwatch
RUN pip install --no-cache-dir '.[service]'
COPY datasets ./datasets
ENV PYTHONUNBUFFERED=1
# Default command is the API; compose overrides it for the worker and mock provider.
CMD ["python", "-m", "shiftwatch.service", "api"]
