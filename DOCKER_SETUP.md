# Docker Setup & Deployment Guide

## 📦 Step 1: Install Docker

### Windows:
1. Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
2. Run the installer
3. Restart your computer when prompted
4. Launch Docker Desktop
5. Verify installation:
   ```powershell
   docker --version
   docker-compose --version
   ```

### Mac:
1. Download [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop)
2. Install and launch Docker Desktop
3. Verify installation:
   ```bash
   docker --version
   docker-compose --version
   ```

### Linux:
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Verify
docker --version
```

---

## 🚀 Step 2: Build Docker Image

Once Docker is installed, run:

```bash
# Navigate to project directory
cd C:\Users\Admin\pechaydetect

# Build the Docker image
docker build -t pechaydetect:latest .
```

**Build time:** ~5-10 minutes (first time, downloads dependencies)

---

## 🏃 Step 3: Run Docker Container

### Option A: Using Docker Compose (Easiest)

1. **Create `.env` file** (optional):
   ```env
   FLASK_SECRET_KEY=your-random-secret-key-here
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_ANON_KEY=your-anon-key
   ```

2. **Run:**
   ```bash
   docker-compose up -d
   ```

3. **View logs:**
   ```bash
   docker-compose logs -f
   ```

4. **Access:** `http://localhost:8080`

### Option B: Using Docker Run

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

---

## ✅ Step 4: Verify Deployment

1. **Check container status:**
   ```bash
   docker ps
   ```

2. **View logs:**
   ```bash
   docker logs -f pechaydetect
   ```

3. **Test health endpoint:**
   ```bash
   curl http://localhost:8080/healthz
   ```

4. **Open in browser:**
   - `http://localhost:8080`

---

## 🛑 Stop/Remove Container

```bash
# Stop container
docker stop pechaydetect

# Remove container
docker rm pechaydetect

# Or using docker-compose
docker-compose down
```

---

## 📋 Quick Reference

### Build Commands
```bash
# Build image
docker build -t pechaydetect:latest .

# Build without cache
docker build --no-cache -t pechaydetect:latest .
```

### Run Commands
```bash
# Run in background
docker run -d -p 8080:8080 --name pechaydetect pechaydetect:latest

# Run with environment variables
docker run -d -p 8080:8080 \
  -e FLASK_SECRET_KEY=secret \
  -e SUPABASE_URL=url \
  -e SUPABASE_ANON_KEY=key \
  pechaydetect:latest
```

### Management Commands
```bash
# View logs
docker logs -f pechaydetect

# Stop
docker stop pechaydetect

# Start
docker start pechaydetect

# Remove
docker rm pechaydetect

# Execute bash in container
docker exec -it pechaydetect bash
```

---

## 🌐 Deploy to Cloud

### DigitalOcean
1. Push image to Docker Hub or DigitalOcean Container Registry
2. Create App Platform → Select Dockerfile
3. Add environment variables
4. Deploy!

### Google Cloud Run
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/pechaydetect
gcloud run deploy pechaydetect --image gcr.io/PROJECT_ID/pechaydetect
```

### AWS ECS/Fargate
1. Push to ECR
2. Create task definition
3. Deploy service

---

## 🔧 Troubleshooting

**Issue: Docker not found**
- Install Docker Desktop
- Restart terminal/computer
- Verify: `docker --version`

**Issue: Port 8080 already in use**
```bash
# Use different port
docker run -p 8081:8080 pechaydetect:latest
```

**Issue: Build fails**
```bash
# Check Dockerfile
# Check requirements.txt
# Try: docker build --no-cache -t pechaydetect:latest .
```

**Issue: Container exits immediately**
```bash
# Check logs
docker logs pechaydetect

# Run interactively
docker run -it pechaydetect:latest
```

---

## 📝 Notes

- **First build takes 5-10 minutes** (downloads all dependencies)
- **Subsequent builds are faster** (uses cache)
- **YOLO is disabled** by default (memory optimization)
- **Uploads persist** via volume mount
- **Health check** runs every 30 seconds

