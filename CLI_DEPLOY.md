# CLI Deployment Guide

Deploy the Pechay Detection System using command-line tools.

---

## 🐳 Option 1: Docker CLI Deployment

### Prerequisites
Install Docker Desktop: https://www.docker.com/products/docker-desktop

### Step 1: Build Docker Image
```bash
cd C:\Users\Admin\pechaydetect
docker build -t pechaydetect:latest .
```

### Step 2: Run Container
```bash
docker run -d \
  --name pechaydetect \
  -p 8080:8080 \
  -e FLASK_SECRET_KEY=your-secret-key \
  -e SUPABASE_URL=https://your-project.supabase.co \
  -e SUPABASE_ANON_KEY=your-anon-key \
  -e FLASK_ENV=production \
  -v ${PWD}/uploads:/app/uploads \
  pechaydetect:latest
```

**Windows PowerShell:**
```powershell
docker run -d `
  --name pechaydetect `
  -p 8080:8080 `
  -e FLASK_SECRET_KEY=your-secret-key `
  -e SUPABASE_URL=https://your-project.supabase.co `
  -e SUPABASE_ANON_KEY=your-anon-key `
  -e FLASK_ENV=production `
  -v ${PWD}/uploads:/app/uploads `
  pechaydetect:latest
```

### Step 3: Check Status
```bash
docker ps
docker logs -f pechaydetect
```

### Step 4: Access App
Open: `http://localhost:8080`

---

## 🚂 Option 2: Railway CLI Deployment

### Install Railway CLI
```bash
# Windows (PowerShell)
iwr https://railway.app/install.ps1 | iex

# Or using npm
npm i -g @railway/cli
```

### Deploy
```bash
# Login
railway login

# Initialize project
railway init

# Link to existing project (optional)
railway link

# Set environment variables
railway variables set FLASK_SECRET_KEY=your-secret-key
railway variables set SUPABASE_URL=https://your-project.supabase.co
railway variables set SUPABASE_ANON_KEY=your-anon-key
railway variables set FLASK_ENV=production
railway variables set ULTRALYTICS_NO_FASTSAM=1

# Deploy
railway up
```

---

## ✈️ Option 3: Fly.io CLI Deployment

### Install Fly CLI
```bash
# Windows (PowerShell)
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"

# Or download from: https://fly.io/docs/getting-started/installing-flyctl/
```

### Deploy
```bash
# Login
fly auth login

# Launch (uses fly.toml)
fly launch

# Set secrets
fly secrets set FLASK_SECRET_KEY=your-secret-key
fly secrets set SUPABASE_URL=https://your-project.supabase.co
fly secrets set SUPABASE_ANON_KEY=your-anon-key
fly secrets set FLASK_ENV=production
fly secrets set ULTRALYTICS_NO_FASTSAM=1

# Deploy
fly deploy

# Open app
fly open
```

---

## ☁️ Option 4: Google Cloud Run CLI

### Prerequisites
```bash
# Install gcloud CLI
# Download from: https://cloud.google.com/sdk/docs/install

# Login
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID
```

### Deploy
```bash
# Build and push
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/pechaydetect

# Deploy
gcloud run deploy pechaydetect \
  --image gcr.io/YOUR_PROJECT_ID/pechaydetect \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars FLASK_SECRET_KEY=your-secret-key,SUPABASE_URL=https://your-project.supabase.co,SUPABASE_ANON_KEY=your-anon-key
```

---

## 🐳 Option 5: Docker Compose CLI

### Deploy with Compose
```bash
# Create .env file (optional)
echo "FLASK_SECRET_KEY=your-secret-key" > .env
echo "SUPABASE_URL=https://your-project.supabase.co" >> .env
echo "SUPABASE_ANON_KEY=your-anon-key" >> .env

# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## 📋 Quick Reference Commands

### Docker Commands
```bash
# Build
docker build -t pechaydetect:latest .

# Run
docker run -d -p 8080:8080 --name pechaydetect pechaydetect:latest

# Logs
docker logs -f pechaydetect

# Stop
docker stop pechaydetect

# Remove
docker rm pechaydetect

# Execute in container
docker exec -it pechaydetect bash
```

### Docker Compose Commands
```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Logs
docker-compose logs -f

# Rebuild
docker-compose up -d --build

# Restart
docker-compose restart
```

---

## 🔧 Troubleshooting

### Docker not found
```bash
# Install Docker Desktop first
# https://www.docker.com/products/docker-desktop
```

### Port already in use
```bash
# Use different port
docker run -p 8081:8080 pechaydetect:latest
```

### Build fails
```bash
# Check Dockerfile
# Try without cache
docker build --no-cache -t pechaydetect:latest .
```

### Container exits immediately
```bash
# Check logs
docker logs pechaydetect

# Run interactively
docker run -it pechaydetect:latest
```

---

## 🎯 Recommended: Docker Compose

**Easiest method:**
```bash
docker-compose up -d
```

This single command:
- Builds the image
- Starts the container
- Sets up volumes
- Configures networking
- Runs in background

---

## 📝 Environment Variables

Required variables:
- `FLASK_SECRET_KEY` - Random secret key
- `SUPABASE_URL` - Your Supabase project URL
- `SUPABASE_ANON_KEY` - Your Supabase anon key
- `FLASK_ENV` - Set to `production`
- `ULTRALYTICS_NO_FASTSAM` - Set to `1` (already in Dockerfile)

---

## ✅ Post-Deployment

1. Check container status: `docker ps`
2. View logs: `docker logs -f pechaydetect`
3. Test health: `curl http://localhost:8080/healthz`
4. Open browser: `http://localhost:8080`

