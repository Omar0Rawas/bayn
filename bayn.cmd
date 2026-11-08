@echo off
rem Bayn task runner entry point for Windows.
rem Works even when PowerShell script execution is disabled (the Windows default):
rem the policy is bypassed for this one call only, nothing on the system is changed.
rem   Usage:  .\bayn setup | demo | run | check | test | doctor
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bayn.ps1" %*
exit /b %ERRORLEVEL%
