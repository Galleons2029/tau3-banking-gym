#!/bin/bash

echo "Starting the Tau3 server..."
uvicorn src.tau3.api_service.simulation_service:app --host 127.0.0.1 --port 8001