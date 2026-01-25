# Docker Deployment Guide

This guide shows how to deploy the Pechay Detection System using Docker.

## 🐳 Prerequisites

- Docker installed ([Download Docker](https://www.docker.com/products/docker-desktop))
- Docker Compose (included with Docker Desktop)

---

## 🚀 Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Create `.env` file** (optional, for environment variables):
   ```bash
   FLASK_SECRET_KEY=your-random-secret-key-here
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_ANON_KEY=your-anon-key
   ```

2. **Build and run:**
   ```bash
   docker-compose up -d
   ```

3. **View logs:**
   ```bash
   docker-compose logs -f
   ```

4. **Stop:**
   ```bash
   docker-compose down
   ```

5. **Access app:**
   - Open browser: `http://localhost:8080`

---

### Option 2: Using Docker Commands

1. **Build the image:**
   ```bash
   docker build -t pechaydetect:latest .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     --name pechaydetect \
     -p 8080:8080 \
     -e FLASK_SECRET_KEY=your-secret-key \
     -e SUPABASE_URL=https://your-project.supabase.co \
     -e SUPABASE_ANON_KEY=your-anon-key \
     -e FLASK_ENV=production \
     -v $(pwd)/uploads:/app/uploads \
     pechaydetect:latest
   ```

3. **View logs:**
   ```bash
   docker logs -f pechaydetect
   ```

4. **Stop:**
   ```bash
   docker stop pechaydetect
   docker rm pechaydetect
   ```

---

## 📋 Environment Variables

Required environment variables:

```bash
FLASK_SECRET_KEY=your-random-secret-key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
FLASK_ENV=production
PORT=8080  # Optional, defaults to 8080
ULTRALYTICS_NO_FASTSAM=1  # Already set in Dockerfile
```

---

## 🔧 Docker Commands Reference

### Build Commands
```bash
# Build image
docker build -t pechaydetect:latest .

# Build without cache
docker build --no-cache -t pechaydetect:latest .

# Build with specific tag
docker build -t pechaydetect:v1.0 .
```

### Run Commands
```bash
# Run in foreground (see logs)
docker run -p 8080:8080 pechaydetect:latest

# Run in background
docker run -d -p 8080:8080 --name pechaydetect pechaydetect:latest

# Run with environment variables
docker run -d -p 8080:8080 \
  -e FLASK_SECRET_KEY=secret \
  -e SUPABASE_URL=url \
  pechaydetect:latest
```

### Management Commands
```bash
# View running containers
docker ps

# View all containers (including stopped)
docker ps -a

# View logs
docker logs pechaydetect
docker logs -f pechaydetect  # Follow logs

# Stop container
docker stop pechaydetect

# Start stopped container
docker start pechaydetect

# Remove container
docker rm pechaydetect

# Remove image
docker rmi pechaydetect:latest

# Execute command in running container
docker exec -it pechaydetect bash
```

### Docker Compose Commands
```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Rebuild and restart
docker-compose up -d --build

# Stop and remove volumes
docker-compose down -v
```

---

## 🌐 Deploy to Cloud Platforms

### DigitalOcean App Platform
1. Connect GitHub repository
2. Select Dockerfile
3. Add environment variables
4. Deploy!

### Google Cloud Run
```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/pechaydetect

# Deploy
gcloud run deploy pechaydetect \
  --image gcr.io/PROJECT_ID/pechaydetect \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### AWS ECS/Fargate
1. Push image to ECR
2. Create ECS task definition
3. Create service
4. Deploy!

### Azure Container Instances
```bash
az container create \
  --resource-group myResourceGroup \
  --name pechaydetect \
  --image pechaydetect:latest \
  --dns-name-label pechaydetect \
  --ports 8080 \
  --environment-variables \
    FLASK_SECRET_KEY=secret \
    SUPABASE_URL=url
```

---

## 🔍 Troubleshooting

### Issue: Container exits immediately
**Solution:**
```bash
# Check logs
docker logs pechaydetect

# Run interactively to see errors
docker run -it pechaydetect:latest
```

### Issue: Port already in use
**Solution:**
```bash
# Use different port
docker run -p 8081:8080 pechaydetect:latest

# Or stop existing container
docker stop pechaydetect
```

### Issue: Permission denied on uploads
**Solution:**
```bash
# Fix permissions
docker exec pechaydetect chmod -R 777 /app/uploads
```

### Issue: Out of memory
**Solution:**
- Increase Docker memory limit (Docker Desktop → Settings → Resources)
- Or use smaller image size in Dockerfile

### Issue: Build fails
**Solution:**
```bash
# Check Dockerfile syntax
docker build --no-cache -t pechaydetect:latest .

# Check requirements.txt
pip install -r requirements.txt
```

---

## 📊 Container Health

### Check health status:
```bash
docker inspect pechaydetect | grep Health -A 10
```

### Health check endpoint:
```bash
curl http://localhost:8080/healthz
```

---

## 🔐 Security Best Practices

1. **Never commit `.env` file** - Use environment variables
2. **Use secrets management** - Docker secrets or cloud secrets
3. **Keep images updated** - Regularly rebuild with latest dependencies
4. **Limit resources** - Set memory/CPU limits
5. **Use non-root user** - Add `USER` directive in Dockerfile (optional)

---

## 📝 Notes

- **YOLO is disabled** by default to prevent memory issues
- **Uploads directory** is persisted via volume mount
- **Health check** runs every 30 seconds
- **Gunicorn** runs with 1 worker, 1 thread (memory optimized)

---

## 🎯 Quick Test

After deployment:
1. Visit: `http://localhost:8080`
2. Login with credentials
3. Upload test image
4. Check detection results

---

## 🔗 Useful Links

- [Docker Documentation](https://docs.docker.com)
- [Docker Compose Documentation](https://docs.docker.com/compose)
- [Docker Hub](https://hub.docker.com)

