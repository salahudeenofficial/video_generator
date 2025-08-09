#!/bin/bash

# Video Generator Update and Sync Script
# This script helps sync changes between local and remote repositories

set -e  # Exit on any error

echo "🔄 Video Generator Update and Sync Script"

# Function to check if we're in a git repository
check_git_repo() {
    if [ ! -d ".git" ]; then
        echo "❌ Error: Not in a git repository"
        echo "Please run this script from the project root directory"
        exit 1
    fi
}

# Function to check git status
check_status() {
    echo "📊 Checking git status..."
    git status --porcelain
}

# Function to add and commit changes
commit_changes() {
    local commit_message="$1"
    if [ -z "$commit_message" ]; then
        commit_message="Update: $(date '+%Y-%m-%d %H:%M:%S')"
    fi
    
    echo "📝 Committing changes: $commit_message"
    git add .
    git commit -m "$commit_message"
}

# Function to push changes
push_changes() {
    echo "🚀 Pushing changes to remote..."
    git push origin main
}

# Function to pull changes
pull_changes() {
    echo "⬇️ Pulling changes from remote..."
    git pull origin main
}

# Function to show recent commits
show_recent_commits() {
    echo "📜 Recent commits:"
    git log --oneline -10
}

# Main script logic
check_git_repo

case "${1:-help}" in
    "status")
        check_status
        ;;
    "commit")
        commit_changes "$2"
        ;;
    "push")
        commit_changes "$2"
        push_changes
        ;;
    "pull")
        pull_changes
        ;;
    "sync")
        echo "🔄 Full sync: pull -> commit -> push"
        pull_changes
        commit_changes "$2"
        push_changes
        ;;
    "log")
        show_recent_commits
        ;;
    "help"|*)
        echo "Usage: $0 [command] [commit_message]"
        echo ""
        echo "Commands:"
        echo "  status                    - Check git status"
        echo "  commit [message]          - Add and commit changes"
        echo "  push [message]            - Commit and push changes"
        echo "  pull                      - Pull latest changes from remote"
        echo "  sync [message]            - Pull, commit, and push (full sync)"
        echo "  log                       - Show recent commits"
        echo "  help                      - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 status                 - Check current status"
        echo "  $0 commit 'Fix bug in pipeline'"
        echo "  $0 push 'Add new feature'"
        echo "  $0 sync 'Update dependencies'"
        ;;
esac 