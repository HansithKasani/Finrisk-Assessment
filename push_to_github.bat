@echo off
REM Script to initialize Git and push to GitHub
REM Repository: FinRisk-Assessment

echo ========================================
echo GitHub Repository Setup
echo ========================================
echo.

REM Check if Git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed or not in PATH
    echo Please install Git from: https://git-scm.com/download/win
    pause
    exit /b 1
)

echo Git is installed. Proceeding with repository setup...
echo.

REM Initialize Git repository
echo [1/7] Initializing Git repository...
git init
if errorlevel 1 (
    echo ERROR: Failed to initialize Git repository
    pause
    exit /b 1
)
echo ✓ Git repository initialized
echo.

REM Add all files
echo [2/7] Adding files to Git...
git add .
if errorlevel 1 (
    echo ERROR: Failed to add files
    pause
    exit /b 1
)
echo ✓ Files added to staging
echo.

REM Create initial commit
echo [3/7] Creating initial commit...
git commit -m "Initial commit: FinRisk-Assessment AI Credit Risk System - Complete implementation with XGBoost, SHAP/LIME, Streamlit dashboard, comprehensive testing, and ethics framework"
if errorlevel 1 (
    echo ERROR: Failed to create commit
    pause
    exit /b 1
)
echo ✓ Initial commit created
echo.

REM Rename branch to main
echo [4/7] Renaming branch to 'main'...
git branch -M main
echo ✓ Branch renamed to 'main'
echo.

REM Add remote repository
echo [5/7] Adding remote repository...
echo.
echo IMPORTANT: You need to create the repository on GitHub first!
echo.
echo Steps to create GitHub repository:
echo 1. Go to: https://github.com/new
echo 2. Repository name: FinRisk-Assessment
echo 3. Description: AI-Powered Credit Risk Assessment System with Explainable AI
echo 4. Visibility: Public
echo 5. DO NOT initialize with README, .gitignore, or license
echo 6. Click "Create repository"
echo.
echo After creating the repository, press any key to continue...
pause >nul

REM Prompt for GitHub username
echo.
set /p GITHUB_USERNAME="Enter your GitHub username: "

echo.
echo Adding remote origin...
git remote add origin https://github.com/%GITHUB_USERNAME%/FinRisk-Assessment.git
if errorlevel 1 (
    echo Warning: Remote might already exist, trying to set URL...
    git remote set-url origin https://github.com/%GITHUB_USERNAME%/FinRisk-Assessment.git
)
echo ✓ Remote repository added
echo.

REM Push to GitHub
echo [6/7] Pushing to GitHub...
echo.
echo You will be prompted for GitHub credentials...
echo (Use Personal Access Token as password if 2FA is enabled)
echo.
git push -u origin main
if errorlevel 1 (
    echo.
    echo ERROR: Failed to push to GitHub
    echo.
    echo Troubleshooting:
    echo 1. Verify repository exists: https://github.com/%GITHUB_USERNAME%/FinRisk-Assessment
    echo 2. Check your GitHub credentials
    echo 3. If using 2FA, create Personal Access Token:
    echo    - Go to: https://github.com/settings/tokens
    echo    - Generate new token (classic)
    echo    - Select 'repo' scope
    echo    - Use token as password
    echo.
    pause
    exit /b 1
)
echo ✓ Code pushed to GitHub
echo.

REM Final message
echo [7/7] Setup complete!
echo.
echo ========================================
echo ✓ SUCCESS!
echo ========================================
echo.
echo Your repository is now available at:
echo https://github.com/%GITHUB_USERNAME%/FinRisk-Assessment
echo.
echo Next steps:
echo 1. Visit your repository on GitHub
echo 2. Add repository description
echo 3. Add topics/tags: machine-learning, credit-risk, xgboost, explainable-ai, streamlit
echo 4. Update README.md with your repository URL
echo 5. Consider adding:
echo    - About section
echo    - Website link (if deployed)
echo    - GitHub Pages for documentation
echo.
pause
