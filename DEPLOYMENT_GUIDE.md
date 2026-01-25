# Deployment Guide - Multiple Platforms

This guide shows how to deploy the Pechay Detection System to various platforms.

## 🚀 Quick Deploy Options

### 1. **Railway** (Recommended - Easy & Free Tier)
1. Go to [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway will auto-detect `railway.json` or `Procfile`
5. Add environment variables:
   - `FLASK_SECRET_KEY` (generate random string)
   - `SUPABASE_URL`
   - `SUPABASE_ANON_KEY`
   - `FLASK_ENV=production`
6. Deploy! Get your domain: `your-app.railway.app`

**Pros:** Free tier, auto-deploy from GitHub, easy setup
**Cons:** Sleeps after inactivity (free tier)

---

### 2. **Fly.io** (Good Performance)
1. Install Fly CLI: `curl -L https://fly.io/install.sh | sh`
2. Login: `fly auth login`
3. Launch: `fly launch` (will use `fly.toml`)
4. Set secrets:
   ```bash
   fly secrets set FLASK_SECRET_KEY=your-secret-key
   fly secrets set SUPABASE_URL=your-url
   fly secrets set SUPABASE_ANON_KEY=your-key
   ```
5. Deploy: `fly deploy`
6. Get domain: `fly open`

**Pros:** Good performance, global CDN, no sleep
**Cons:** Requires CLI setup

---

### 3. **Docker/Any Platform** (Universal)
The `Dockerfile` works on:
- **DigitalOcean App Platform**
- **AWS ECS/Fargate**
- **Google Cloud Run**
- **Azure Container Instances**
- **Heroku** (with Container Registry)

**DigitalOcean Example:**
1. Go to [DigitalOcean App Platform](https://cloud.digitalocean.com/apps)
2. Create App → GitHub → Select repo
3. Auto-detects Dockerfile
4. Add environment variables
5. Deploy!

**Google Cloud Run:**
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/detectpechay
gcloud run deploy detectpechay --image gcr.io/PROJECT_ID/detectpechay --platform managed
```

---

### 4. **Heroku** (Classic Platform)
1. Install Heroku CLI
2. Login: `heroku login`
3. Create app: `heroku create your-app-name`
4. Set config vars:
   ```bash
   heroku config:set FLASK_SECRET_KEY=your-key
   heroku config:set SUPABASE_URL=your-url
   heroku config:set SUPABASE_ANON_KEY=your-key
   ```
5. Deploy: `git push heroku main`

**Note:** Heroku uses `Procfile` automatically

---

### 5. **Vercel** (Serverless - Requires Changes)
Vercel is serverless, so Flask needs adaptation:
- Use `vercel.json` with serverless functions
- Or use Vercel's Python runtime

**Not recommended** for this app due to memory requirements.

---

## 📋 Environment Variables Needed

All platforms need these:
```bash
FLASK_SECRET_KEY=your-random-secret-key-here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
FLASK_ENV=production
PYTHON_VERSION=3.11
ULTRALYTICS_NO_FASTSAM=1
```

---

## 🔧 Platform-Specific Notes

### Railway
- Uses `railway.json` or `Procfile`
- Free tier: 500 hours/month
- Auto-deploys from GitHub

### Fly.io
- Uses `fly.toml`
- Free tier: 3 shared VMs
- Global edge network

### Docker
- Works everywhere Docker runs
- Most flexible option
- Use `Dockerfile` provided

### Render (Current)
- Uses `render.yaml` or `Procfile`
- Free tier available
- Auto-deploy from GitHub

---

## 🎯 Recommended: Railway

**Why Railway?**
- ✅ Easiest setup
- ✅ Free tier (500 hours/month)
- ✅ Auto-deploy from GitHub
- ✅ Good documentation
- ✅ No credit card required (free tier)

**Steps:**
1. Sign up at [railway.app](https://railway.app)
2. New Project → GitHub
3. Select repo → Deploy
4. Add environment variables
5. Done! 🎉

---

## 📝 Post-Deployment Checklist

- [ ] Test login functionality
- [ ] Test image upload
- [ ] Verify detection works
- [ ] Check session persistence
- [ ] Test all dashboard pages
- [ ] Verify environment variables are set
- [ ] Check logs for errors

---

## 🆘 Troubleshooting

**Memory Issues:**
- Ensure `ULTRALYTICS_NO_FASTSAM=1` is set
- Check worker count (should be 1)
- Verify timeout settings

**Session Issues:**
- Check `FLASK_SECRET_KEY` is set
- Verify `SESSION_COOKIE_SECURE` matches platform (HTTPS/HTTP)

**Import Errors:**
- Check all dependencies in `requirements.txt`
- Verify Python version (3.11)

---

## 🔗 Quick Links

- [Railway Docs](https://docs.railway.app)
- [Fly.io Docs](https://fly.io/docs)
- [Docker Docs](https://docs.docker.com)
- [Render Docs](https://render.com/docs)

