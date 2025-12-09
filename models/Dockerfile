FROM python:3.12-slim

WORKDIR /app

# 1. Install System Dependencies (Fixes the libgomp.so.1 OSError)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# 2. Copy and Install Python Dependencies
COPY minimal_requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 3. API dependencies (Combined with requirements.txt, or you can keep this separate if preferred)
# Since fastapi and uvicorn were likely already in your requirements.txt from the previous build attempts,
# I'll rely on the main requirements file for everything.

# If you prefer to keep them separate for clarity, the file is fine as is.
# If they are NOT in your requirements file, you would add:
# RUN pip install fastapi uvicorn

# 4. Copy Application Files
COPY flaml_balanced_sentiment_pipeline.joblib .
COPY api.py .

# 5. Expose Port and Define Command
EXPOSE 8000

# Correct Uvicorn syntax
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]