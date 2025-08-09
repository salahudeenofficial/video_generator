# GitHub Repository Setup Guide

This guide will help you set up a GitHub repository for the Video Generator project and manage the workflow between your local machine and VAST AI instances.

## 🚀 **Step 1: Create GitHub Repository**

1. **Go to GitHub.com** and sign in to your account
2. **Click "New repository"** or the "+" icon
3. **Repository name**: `video-generator`
4. **Description**: `A comprehensive video generation pipeline using VACE and Wan2.1`
5. **Visibility**: Choose Public or Private
6. **Initialize with**: 
   - ✅ Add a README file
   - ✅ Add .gitignore (choose Python template)
   - ✅ Choose a license (MIT recommended)
7. **Click "Create repository"**

## 🔧 **Step 2: Local Repository Setup**

```bash
# Navigate to your project directory
cd ~/video_generator

# Initialize git repository (if not already done)
git init

# Add remote origin
git remote add origin https://github.com/YOUR_USERNAME/video-generator.git

# Add all files
git add .

# Initial commit
git commit -m "Initial commit: Video Generator with VACE and Wan2.1"

# Push to GitHub
git push -u origin main
```

## 📁 **Step 3: Repository Structure**

Your repository should now contain:
```
video-generator/
├── .gitignore              # Excludes unnecessary files
├── setup.py                # Package installation
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── scripts/               # Utility scripts
│   ├── setup_vast_ai.sh   # VAST AI setup script
│   ├── install_dependencies.sh # Dependency installer
│   └── update_and_sync.sh # Sync helper
├── video_generator/       # Main package
│   ├── __init__.py
│   ├── r2v_pipeline.py
│   ├── config.py
│   └── utils.py
├── examples/              # Usage examples
├── test_setup.py         # Setup verification
└── VACE/                 # VACE framework (submodule)
```

## 🔄 **Step 4: Workflow Management**

### **Local Development Workflow:**
```bash
# Make changes to your code
# Test locally
python test_setup.py

# Commit and push changes
./scripts/update_and_sync.sh push "Your commit message"

# Or use individual commands:
git add .
git commit -m "Your commit message"
git push origin main
```

### **VAST AI Instance Workflow:**
```bash
# On VAST AI instance
./scripts/setup_vast_ai.sh

# To get latest changes
git pull origin main

# To install/update dependencies
./scripts/install_dependencies.sh
```

## 📋 **Step 5: Daily Workflow**

### **Starting Work (Local):**
```bash
# Pull latest changes
./scripts/update_and_sync.sh pull

# Make your changes
# Test your changes
python test_setup.py
```

### **Finishing Work (Local):**
```bash
# Commit and push changes
./scripts/update_and_sync.sh push "Description of changes made"
```

### **On VAST AI:**
```bash
# Get latest changes
git pull origin main

# Install any new dependencies
./scripts/install_dependencies.sh

# Run your video generation
python examples/r2v_example.py
```

## 🛠️ **Step 6: Advanced Git Operations**

### **Check Status:**
```bash
./scripts/update_and_sync.sh status
```

### **View Recent Commits:**
```bash
./scripts/update_and_sync.sh log
```

### **Full Sync (Pull + Commit + Push):**
```bash
./scripts/update_and_sync.sh sync "Your message"
```

## 🔍 **Step 7: Troubleshooting**

### **Merge Conflicts:**
```bash
# If you get merge conflicts
git status                    # See conflicted files
# Edit conflicted files manually
git add .                     # Add resolved files
git commit -m "Resolve conflicts"
git push origin main
```

### **Reset to Remote:**
```bash
# If you want to discard local changes and use remote version
git fetch origin
git reset --hard origin/main
```

### **Check Remote Status:**
```bash
git remote -v                 # Check remote URLs
git branch -a                 # See all branches
```

## 📱 **Step 8: GitHub Actions (Optional)**

You can set up GitHub Actions for automated testing:

1. **Create `.github/workflows/test.yml`**
2. **Configure automated testing on push/pull requests**
3. **Set up dependency caching**

## 🎯 **Best Practices**

1. **Commit Frequently**: Small, focused commits are better than large ones
2. **Use Descriptive Messages**: Explain what and why, not how
3. **Test Before Pushing**: Always test locally before pushing
4. **Pull Before Push**: Always pull latest changes before pushing
5. **Use Branches**: Create feature branches for major changes
6. **Document Changes**: Update README.md when adding new features

## 🚨 **Important Notes**

- **Never commit large files** (models, videos, etc.) - they're in `.gitignore`
- **Keep VACE as a submodule** or document how to clone it separately
- **Update requirements.txt** when adding new dependencies
- **Test on VAST AI** before marking features as complete

## 📞 **Need Help?**

- Check the main `README.md` for project details
- Use `./scripts/update_and_sync.sh help` for script usage
- Review git logs with `./scripts/update_and_sync.sh log`
- Check GitHub Issues for known problems

---

**Happy coding! 🎉** 