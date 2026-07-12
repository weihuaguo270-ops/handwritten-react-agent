@echo off
REM react-setup.bat — 将 react CLI 添加到系统 PATH
REM 以管理员身份运行一次即可

set "TARGET_DIR=%~dp0"
set "CURRENT_PATH=%PATH%"

echo "%CURRENT_PATH%" | find /i "%TARGET_DIR%" >nul
if %errorlevel% equ 0 (
    echo react 已在 PATH 中
) else (
    echo 正在将 %TARGET_DIR% 添加到用户 PATH...
    setx PATH "%TARGET_DIR%;%PATH%"
    echo 完成！请重新打开终端后即可直接使用 react
)
echo.
echo 使用方法:
echo   react                   启动交互模式
echo   react run "你的问题"    单次执行
echo   react config            查看配置
