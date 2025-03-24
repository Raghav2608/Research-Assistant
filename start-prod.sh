#!/bin/bash
# Start frontend service
npm run build --prefix ./frontend
npm run start --prefix ./frontend&

# Wait until the frontend is available on port 3000
while ! nc -z localhost 3000; do   
  echo "Waiting for frontend to start..."
  sleep 1
done

echo "Frontend is up. Starting backend services..."
# Start backend services on different ports
uvicorn backend.apps.app_webapp:app --host 0.0.0.0 --port 8000 &
uvicorn backend.apps.app_data_ingestion:app --host 0.0.0.0 --port 8001 &
uvicorn backend.apps.app_retrieval:app --host 0.0.0.0 --port 8002 &
uvicorn backend.apps.app_llm_inference:app --host 0.0.0.0 --port 8003 &
# Wait for all background processes to exit
wait
