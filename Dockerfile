FROM python:3.10-slim

WORKDIR /app

# Copy requirements from root to container
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

# Copy the rest of the application
COPY . .

# Run the application from the app directory
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]

