@echo off
chcp 65001 >nul
cd /d "%~dp0"
python terminal_chat_qwen3_python.py --device cpu
