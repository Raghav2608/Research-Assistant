uvicorn backend.apps.app_webapp:app --host 0.0.0.0 --port 8000 &
uvicorn backend.apps.app_data_ingestion:app --host 0.0.0.0 --port 8001 &
uvicorn backend.apps.app_retrieval:app --host 0.0.0.0 --port 8002 &
uvicorn backend.apps.app_llm_inference:app --host localhost --port 8003 &
npm run dev --prefix ./frontend &
wait  # Ensures the script waits for all processes
