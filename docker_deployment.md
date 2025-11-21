# Dockerfile and Deployment Guide for `ai_drug_summary`

This guide provides a production-ready Dockerfile and instructions for building, running, and deploying the `ai_drug_summary` Python module.

---

## 1. Directory Structure

A recommended project layout:

```
ai_drug_summary_project/
│
├── ai_drug_summary.py
├── requirements.txt
├── Dockerfile
├── sample_medications.csv
└── README.md
```

---

## 2. `requirements.txt`

Example minimal dependencies:

```
pandas
pydantic
```

If you later integrate a real LLM client, add its SDK here.

---

## 3. Dockerfile

```dockerfile
# ---- Stage 1: Base image ----
FROM python:3.11-slim AS base

# Install system dependencies (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ai_drug_summary.py ./

# Default command – override via CLI
ENTRYPOINT ["python", "ai_drug_summary.py"]
```

---

## 4. Building the Docker Image

Build the container from the project root:

```bash
docker build -t ai_drug_summary:latest .
```

---

## 5. Running the Container

Run with a mounted volume for input and output files:

```bash
docker run --rm \
  -v $(pwd):/data \
  ai_drug_summary:latest \
  --input /data/sample_medications.csv \
  --output /data/output_summary.json
```

Explanation:
- `--rm` removes the container after execution.
- `-v $(pwd):/data` mounts your host directory into `/data` inside the container.
- The script reads the CSV and writes the JSON summary.

---

## 6. Environment Variables (Optional)

If using a real LLM provider:

```
ENV LLM_PROVIDER=openai
ENV OPENAI_API_KEY=your_key_here
```

Inject securely at runtime:

```bash
docker run --rm \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd):/data \
  ai_drug_summary:latest \
  --input /data/sample_medications.csv
```

---

## 7. Deploying to a Cloud Platform

### **Amazon ECS / Fargate**
1. Push the image:
   ```bash
   docker tag ai_drug_summary your_repo/ai_drug_summary
   docker push your_repo/ai_drug_summary
   ```
2. Create an ECS Task Definition with:
   - CPU/memory (light usage)
   - Mounted S3 bucket or EFS for input/output
   - Secrets injected via AWS Secrets Manager

3. Run tasks on demand or schedule with EventBridge.

### **Google Cloud Run**
Cloud Run is ideal for on‑demand processing.

1. Submit build:
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT/ai_drug_summary
   ```
2. Deploy:
   ```bash
   gcloud run deploy ai-drug-summary \
      --image gcr.io/PROJECT/ai_drug_summary \
      --region us-central1 \
      --memory 512Mi
   ```
3. Invoke via authenticated request with an uploaded CSV.

### **Azure Container Apps**
1. Push to Azure Container Registry.
2. Deploy a Container App with:
   - Environment variables for LLM provider
   - Azure Blob Storage for file I/O

---

## 8. Scaling Considerations

- The tool is **CPU-bound** and lightweight.
- Memory usage is low (<200MB typical).
- GPU acceleration is not required unless using an on‑prem LLM.
- Stateless: containers can be horizontally scaled.

---

## 9. Production Hardening

- Add request validation and schema checking.
- Log to STDOUT/STDERR only; external aggregator collects logs.
- Run as a non-root user inside Docker for security.
- Use a real LLM provider implementation instead of the dummy client.

---

## 10. Next Steps

I can also provide:
- a Kubernetes deployment manifest
- a serverless batch-processing pipeline
- a FastAPI wrapper to expose this as a web service
- GitHub Actions CI/CD pipeline templates

Let me know what you'd like next.

