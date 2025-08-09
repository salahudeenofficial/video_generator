#!/bin/bash

# GitHub Repository Setup Script
# This script helps set up the local git repository and prepare for GitHub

set -e  # Exit on any error

echo "🚀 Setting up GitHub Repository for Video Generator..."

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "video_generator" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Error: Git is not installed. Please install git first."
    exit 1
fi

# Check if git repository already exists
if [ -d ".git" ]; then
    echo "⚠️ Git repository already exists"
    echo "Current remote:"
    git remote -v
    echo ""
    read -p "Do you want to continue with setup? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Initialize git repository if needed
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
fi

# Get GitHub username
echo "🔗 GitHub Repository Setup"
echo "Please provide your GitHub username:"
read -p "GitHub Username: " github_username

if [ -z "$github_username" ]; then
    echo "❌ Error: GitHub username is required"
    exit 1
fi

# Set up remote origin
echo "🌐 Setting up remote origin..."
if git remote get-url origin &> /dev/null; then
    echo "Remote origin already exists. Updating..."
    git remote set-url origin "https://github.com/$github_username/video_generator.git"
else
    git remote add origin "https://github.com/$github_username/video_generator.git"
fi

# Add all files
echo "📝 Adding files to git..."
git add .

# Initial commit
echo "💾 Creating initial commit..."
git commit -m "Initial commit: Video Generator with VACE and Wan2.1"

# Set main as default branch
git branch -M main

echo ""
echo "✅ Local git repository setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Go to https://github.com/new"
echo "2. Create repository named 'video-generator'"
echo "3. Make it Public or Private as you prefer"
echo "4. DO NOT initialize with README, .gitignore, or license (we already have them)"
echo "5. Click 'Create repository'"
echo ""
echo "🚀 After creating the repository, run:"
echo "  git push -u origin main"
echo ""
echo "🔄 For future updates, use:"
echo "  ./scripts/update_and_sync.sh push 'Your message'"
echo ""
echo "📖 See GITHUB_SETUP.md for detailed workflow instructions" 