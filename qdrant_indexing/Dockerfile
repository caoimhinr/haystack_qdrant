FROM deepset/hayhooks:v0.6.0

# Install system Git CLI
USER root
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir GitPython qdrant-haystack sentence-transformers

# Switch back to default user
USER hayhooks

# Run Hayhooks server
CMD ["hayhooks", "run", "--host", "0.0.0.0"]
