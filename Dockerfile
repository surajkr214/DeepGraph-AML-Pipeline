# 1. Use a lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file
COPY requirements.txt .

# 4. Install dependencies
# We explicitly install the CPU version of PyTorch to keep the image small
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install torch_geometric -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# 5. Copy your application code into the container
COPY . .

# 6. Expose the port the app runs on
EXPOSE 8000

# 7. Define the command to run the app
CMD ["uvicorn", "api_predictor:app", "--host", "0.0.0.0", "--port", "8000"]