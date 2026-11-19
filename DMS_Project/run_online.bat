@echo off
title DMS AI System - Cloudflare Tunnel Launcher
color 0A

:: Xóa sạch log rác C++
set TF_CPP_MIN_LOG_LEVEL=3
set GLOG_minloglevel=3
set ABSL_LOG_SEVERITY=fatal

echo ====================================================================
echo    HE THONG GIAM SAT TAI XE (DMS) - KHOI DONG TU DONG CLOUDFLARE
echo ====================================================================
echo.

:: 1. Mo Cloudflare Tunnel o mot cua so doc lap
echo [1/2] Dang mo duong truyen Cloudflare Tunnel...
start "CLOUDFLARE TUNNEL (LAY LINK O DAY)" cmd /k ".\cloudflared.exe tunnel --url http://127.0.0.1:8501"

:: 2. Doi 2 giay de Cloudflare san sang
timeout /t 2 /nobreak >nul

:: 3. Khoi dong Streamlit Web App
echo [2/2] Dang khoi dong Streamlit Web Server...
streamlit run app_streamlit.py

pause