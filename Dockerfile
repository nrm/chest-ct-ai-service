# =============================================================================
# Stage 1: System dependencies and Python packages
# =============================================================================
FROM python:3.11-slim AS dependencies

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies (this layer will be cached)
# Install heavy, less frequently changed Python packages first
COPY requirements.heavy.txt .
RUN pip install --no-cache-dir -r requirements.heavy.txt

# Install lighter, more frequently changed packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Application code
# =============================================================================
FROM dependencies AS application

# Copy application source code (frequently changing files)
COPY main.py .
COPY covid19/ ./covid19/
COPY hackathon/ ./hackathon/
COPY radiassist/ ./radiassist/

# Copy utility modules
COPY utils/ ./utils/

# Copy segmentation modules
COPY segment_and_viz_2/ ./segment_and_viz_2/

# =============================================================================
# Stage 3: Model weights (large files, change rarely)
# =============================================================================
FROM application AS models

# Copy model weights (large files, cached separately)
COPY models/ ./models/

# =============================================================================
# Stage 4: Runtime configuration
# =============================================================================
FROM models AS runtime

# Create directories for runtime data
RUN mkdir -p /tmp/radiassist /app/data /app/logs

# Set Python path for local modules
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "main.py"]