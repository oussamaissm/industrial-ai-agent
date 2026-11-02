FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -e .

COPY . .

RUN mkdir -p data experiments reports models
RUN python -m data.generate_factory_data

CMD ["python", "-m", "src.evaluation.runner"]
