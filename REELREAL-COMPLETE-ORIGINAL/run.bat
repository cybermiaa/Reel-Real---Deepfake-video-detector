@echo off
cd server
set PIPELINE_DIR=..\ctf_pretrained
python -m uvicorn app:app --host 127.0.0.1 --port 8000
pause