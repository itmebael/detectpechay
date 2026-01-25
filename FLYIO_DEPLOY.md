# Fly.io CLI Deployment Guide

## ⚠️ Important: Payment Required

Fly.io requires a credit card to create apps (even for free tier). The free tier includes:
- 3 shared VMs
- 160GB outbound data transfer/month
- Free SSL certificates

**Add payment info:** https://fly.io/dashboard/abel-redoblado/billing

---

## 🚀 Deployment Steps

### Step 1: Add Payment Information
1. Go to: https://fly.io/dashboard/abel-redoblado/billing
2. Add credit card (required even for free tier)
3. Return to terminal

### Step 2: Create App
```bash
# Add Fly CLI to PATH (Windows PowerShell)
$env:PATH += ";C:\Users\Admin\.fly\bin"

# Create app
flyctl apps create detectpechay --org personal
```

### Step 3: Set Environment Variables (Secrets)
```bash
flyctl secrets set \
  FLASK_SECRET_KEY="your-random-secret-key" \
  SUPABASE_URL="https://your-project.supabase.co" \
  SUPABASE_ANON_KEY="your-anon-key" \
  FLASK_ENV="production" \
  ULTRALYTICS_NO_FASTSAM="1"
```

### Step 4: Deploy
```bash
flyctl deploy
```

### Step 5: Open App
```bash
flyctl open
```

---

## 📋 Complete Deployment Commands

```bash
# 1. Add Fly CLI to PATH (Windows PowerShell)
$env:PATH += ";C:\Users\Admin\.fly\bin"

# 2. Login (if not already)
flyctl auth login

# 3. Create app
flyctl apps create detectpechay --org personal

# 4. Set secrets
flyctl secrets set FLASK_SECRET_KEY="your-secret-key"
flyctl secrets set SUPABASE_URL="https://your-project.supabase.co"
flyctl secrets set SUPABASE_ANON_KEY="your-anon-key"
flyctl secrets set FLASK_ENV="production"
flyctl secrets set ULTRALYTICS_NO_FASTSAM="1"

# 5. Deploy
flyctl deploy

# 6. Open app
flyctl open

# 7. View logs
flyctl logs

# 8. Check status
flyctl status
```

---

## 🔧 Configuration

### fly.toml Settings
- **Memory:** 2048MB (2GB) - optimized for ML models
- **CPU:** 1 shared CPU
- **Region:** iad (Washington, D.C.)
- **Port:** 8080
- **Auto-scaling:** Enabled (stops when idle, starts on request)

### Update Memory (if needed)
```bash
# Scale memory
flyctl scale memory 2048

# Or edit fly.toml and redeploy
```

---

## 📊 Useful Commands

```bash
# View app status
flyctl status

# View logs
flyctl logs

# SSH into VM
flyctl ssh console

# Scale resources
flyctl scale memory 2048
flyctl scale count 1

# View secrets
flyctl secrets list

# Update secrets
flyctl secrets set KEY=value

# Remove secrets
flyctl secrets unset KEY

# Open app in browser
flyctl open

# View app info
flyctl info
```

---

## 🔍 Troubleshooting

### Issue: Payment Required
**Solution:** Add credit card at https://fly.io/dashboard/billing

### Issue: App Not Found
**Solution:** Create app first:
```bash
flyctl apps create detectpechay --org personal
```

### Issue: Memory Errors
**Solution:** Increase memory:
```bash
flyctl scale memory 2048
```

### Issue: Build Fails
**Solution:** Check Dockerfile and requirements.txt
```bash
# Test locally first
docker build -t test .
```

### Issue: Deployment Timeout
**Solution:** Check logs:
```bash
flyctl logs
```

---

## 🌐 Alternative: Railway (No Credit Card for Free Tier)

If you prefer not to add payment info, consider **Railway**:

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Deploy
railway up
```

Railway free tier doesn't require credit card initially.

---

## ✅ Post-Deployment Checklist

- [ ] App created successfully
- [ ] Secrets configured
- [ ] Deployment completed
- [ ] App accessible via `flyctl open`
- [ ] Health check passes: `curl https://detectpechay.fly.dev/healthz`
- [ ] Login works
- [ ] Image upload works
- [ ] Detection works

---

## 🔗 Useful Links

- [Fly.io Dashboard](https://fly.io/dashboard)
- [Fly.io Billing](https://fly.io/dashboard/billing)
- [Fly.io Docs](https://fly.io/docs)
- [Fly.io Pricing](https://fly.io/pricing)

---

## 📝 Notes

- **Free tier** includes 3 shared VMs
- **Auto-scaling** stops VMs when idle (saves resources)
- **First request** may be slow (cold start)
- **Memory** set to 2048MB for ML models
- **YOLO** is disabled by default (memory optimization)

---

## 🎯 Next Steps

1. **Add payment info** at https://fly.io/dashboard/billing
2. **Run deployment commands** above
3. **Test your app** at the provided URL
4. **Monitor usage** in Fly.io dashboard

