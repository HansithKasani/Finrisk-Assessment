"""
Script to initialize Git repository and push to GitHub
Repository: FinRisk-Assessment
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command and return result"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr


def check_git_installed():
    """Check if Git is installed"""
    success, stdout, stderr = run_command("git --version", check=False)
    if not success:
        print("❌ ERROR: Git is not installed or not in PATH")
        print("Please install Git from: https://git-scm.com/downloads")
        return False
    print(f"✓ Git is installed: {stdout.strip()}")
    return True


def initialize_repository():
    """Initialize Git repository"""
    print("\n[1/7] Initializing Git repository...")
    
    # Check if already initialized
    if Path(".git").exists():
        print("⚠️  Git repository already initialized")
        return True
    
    success, stdout, stderr = run_command("git init")
    if not success:
        print(f"❌ ERROR: Failed to initialize repository: {stderr}")
        return False
    
    print("✓ Git repository initialized")
    return True


def add_files():
    """Add all files to Git"""
    print("\n[2/7] Adding files to Git...")
    
    success, stdout, stderr = run_command("git add .")
    if not success:
        print(f"❌ ERROR: Failed to add files: {stderr}")
        return False
    
    print("✓ Files added to staging")
    return True


def create_initial_commit():
    """Create initial commit"""
    print("\n[3/7] Creating initial commit...")
    
    commit_message = (
        "Initial commit: FinRisk-Assessment AI Credit Risk System - "
        "Complete implementation with XGBoost, SHAP/LIME, Streamlit dashboard, "
        "comprehensive testing, and ethics framework"
    )
    
    success, stdout, stderr = run_command(f'git commit -m "{commit_message}"')
    if not success:
        if "nothing to commit" in stderr:
            print("⚠️  No changes to commit")
            return True
        print(f"❌ ERROR: Failed to create commit: {stderr}")
        return False
    
    print("✓ Initial commit created")
    return True


def rename_branch():
    """Rename branch to main"""
    print("\n[4/7] Renaming branch to 'main'...")
    
    success, stdout, stderr = run_command("git branch -M main")
    if not success:
        print(f"⚠️  Warning: {stderr}")
    else:
        print("✓ Branch renamed to 'main'")
    return True


def add_remote_repository():
    """Add remote GitHub repository"""
    print("\n[5/7] Adding remote repository...")
    print("\n" + "="*70)
    print("IMPORTANT: You need to create the repository on GitHub first!")
    print("="*70)
    print("\nSteps to create GitHub repository:")
    print("1. Go to: https://github.com/new")
    print("2. Repository name: FinRisk-Assessment")
    print("3. Description: AI-Powered Credit Risk Assessment System with Explainable AI")
    print("4. Visibility: Public")
    print("5. DO NOT initialize with README, .gitignore, or license")
    print("6. Click 'Create repository'")
    print("\n" + "="*70 + "\n")
    
    input("After creating the repository, press Enter to continue...")
    
    github_username = input("\nEnter your GitHub username: ").strip()
    
    if not github_username:
        print("❌ ERROR: GitHub username is required")
        return False, None
    
    remote_url = f"https://github.com/{github_username}/FinRisk-Assessment.git"
    
    # Try to add remote
    success, stdout, stderr = run_command(f"git remote add origin {remote_url}", check=False)
    
    if not success and "remote origin already exists" in stderr:
        print("⚠️  Remote already exists, updating URL...")
        success, stdout, stderr = run_command(f"git remote set-url origin {remote_url}")
    
    if not success:
        print(f"❌ ERROR: Failed to add remote: {stderr}")
        return False, github_username
    
    print(f"✓ Remote repository added: {remote_url}")
    return True, github_username


def push_to_github():
    """Push code to GitHub"""
    print("\n[6/7] Pushing to GitHub...")
    print("\nℹ️  You will be prompted for GitHub credentials...")
    print("(Use Personal Access Token as password if 2FA is enabled)")
    print()
    
    success, stdout, stderr = run_command("git push -u origin main", check=False)
    
    if not success:
        print("\n❌ ERROR: Failed to push to GitHub")
        print("\nTroubleshooting:")
        print("1. Verify repository exists on GitHub")
        print("2. Check your GitHub credentials")
        print("3. If using 2FA, create Personal Access Token:")
        print("   - Go to: https://github.com/settings/tokens")
        print("   - Generate new token (classic)")
        print("   - Select 'repo' scope")
        print("   - Use token as password")
        print(f"\nError details: {stderr}")
        return False
    
    print("✓ Code pushed to GitHub successfully!")
    return True


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("GitHub Repository Setup - FinRisk-Assessment")
    print("="*70 + "\n")
    
    # Check Git installation
    if not check_git_installed():
        sys.exit(1)
    
    # Initialize repository
    if not initialize_repository():
        sys.exit(1)
    
    # Add files
    if not add_files():
        sys.exit(1)
    
    # Create initial commit
    if not create_initial_commit():
        sys.exit(1)
    
    # Rename branch
    if not rename_branch():
        sys.exit(1)
    
    # Add remote repository
    success, github_username = add_remote_repository()
    if not success:
        sys.exit(1)
    
    # Push to GitHub
    if not push_to_github():
        sys.exit(1)
    
    # Success message
    print("\n[7/7] Setup complete!")
    print("\n" + "="*70)
    print("✓ SUCCESS!")
    print("="*70)
    print(f"\nYour repository is now available at:")
    print(f"https://github.com/{github_username}/FinRisk-Assessment")
    print("\nNext steps:")
    print("1. Visit your repository on GitHub")
    print("2. Add repository description")
    print("3. Add topics/tags: machine-learning, credit-risk, xgboost,")
    print("   explainable-ai, streamlit, python, data-science")
    print("4. Update README.md with your repository URL")
    print("5. Consider adding:")
    print("   - About section")
    print("   - Website link (if deployed)")
    print("   - GitHub Pages for documentation")
    print("   - Star the repository ⭐")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
