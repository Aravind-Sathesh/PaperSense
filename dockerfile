# --- STAGE 1: Build Stage ---
FROM python:3.11-slim AS builder

WORKDIR /app

# --- THIS IS THE FIX ---
# Update the package list AND upgrade all installed packages to their latest versions
# This patches known vulnerabilities in the base OS.
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
# --- END OF FIX ---

COPY requirements.txt .

RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# --- STAGE 2: Final Stage ---
# We also need to update the final stage's OS
FROM python:3.11-slim

# --- THIS IS THE FIX (APPLIED TO THE FINAL STAGE) ---
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*
# --- END OF FIX ---

WORKDIR /app

COPY --from=builder /app/venv ./venv
COPY . .

ENV PATH="/app/venv/bin:$PATH"

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]