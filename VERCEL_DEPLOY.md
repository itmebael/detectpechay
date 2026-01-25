# Deploying to Vercel

This guide will help you deploy the Pechay Detection System to Vercel.

## ⚠️ Important Notes

**Vercel Limitations:**
- Serverless functions have a 60-second timeout (configurable up to 300s on Pro plan)
- File system is read-only except `/tmp` directory
- Memory limit: 2048MB free tier, 3008MB Pro plan
- Cold starts can be slow for heavy ML models (YOLO loading)
- Free tier has usage limits (100GB bandwidth/month)

**Recommendations:**
- Consider Railway or Fly.io for better ML model support
- Vercel works but may have performance issues with YOLO model loading

---

## 🚀 Quick Deploy Steps

### Option 1: Deploy via Vercel Dashboard (Easiest)

1. **Sign up/Login to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Sign up with GitHub

2. **Import Project**
   - Click "Add New..." → "Project"
   - Import your GitHub repository
   - Select the `detectpechay` repository

3. **Configure Project**
   - Framework Preset: **Other** (or leave auto-detected)
   - Root Directory: `./` (root)
   - Build Command: Leave empty (Vercel auto-detects)
   - Output Directory: Leave empty
   - Install Command: `pip install -r requirements.txt`

4. **Set Environment Variables**
   Click "Environment Variables" and add:
   ```
   FLASK_SECRET_KEY=your-random-secret-key-here
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_ANON_KEY=your-anon-key
   FLASK_ENV=production
   PYTHON_VERSION=3.11
   ULTRALYTICS_NO_FASTSAM=1
   ```

5. **Deploy**
   - Click "Deploy"
   - Wait for build to complete
   - Your app will be live at: `https://your-app.vercel.app`

---

### Option 2: Deploy via Vercel CLI

1. **Install Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **Login**
   ```bash
   vercel login
   ```

3. **Deploy (Preview)**
   ```bash
   vercel
   ```
   This creates a preview deployment. Answer the prompts:
   - Link to existing project? **Yes** (if already linked) or **No** (to create new)
   - Project name: `detectpechay` (or your preferred name)
   - Directory: `./` (current directory)
   - Override settings? **No** (use vercel.json)

4. **Set Environment Variables**
   ```bash
   # Interactive mode (will prompt for values)
   vercel env add FLASK_SECRET_KEY
   vercel env add SUPABASE_URL
   vercel env add SUPABASE_ANON_KEY
   vercel env add FLASK_ENV production
   vercel env add PYTHON_VERSION 3.11
   vercel env add ULTRALYTICS_NO_FASTSAM 1
   ```
   
   **Or set all at once:**
   ```bash
   vercel env add FLASK_SECRET_KEY production
   vercel env add SUPABASE_URL production
   vercel env add SUPABASE_ANON_KEY production
   vercel env add FLASK_ENV production production
   vercel env add PYTHON_VERSION production 3.11
   vercel env add ULTRALYTICS_NO_FASTSAM production 1
   ```
   
   **Note:** Replace `production` with `preview` or `development` if needed for different environments.

5. **Deploy to Production**
   ```bash
   vercel --prod
   ```

---

## 📋 Configuration Files

### `vercel.json`
Already configured with:
- Python runtime (`@vercel/python`) via `builds` property
- Routes all requests to `app.py`
- Production environment variables (`FLASK_ENV`, `ULTRALYTICS_NO_FASTSAM`)

**Note:** The current configuration uses `builds` (legacy format) which works reliably with Flask apps. The `functions` property is not used to avoid conflicts.

### File Upload Handling
The app automatically detects Vercel environment and uses `/tmp` directory for file uploads (Vercel's writable directory).

---

## 🔧 Troubleshooting

### Issue: Function Timeout
**Solution:** 
- Default timeout is 60 seconds (free tier)
- To increase timeout, you need Vercel Pro plan
- Update `vercel.json` to use `functions` instead of `builds`:
```json
{
  "version": 2,
  "functions": {
    "app.py": {
      "runtime": "@vercel/python",
      "maxDuration": 300,  // 5 minutes (Pro plan only)
      "memory": 2048  // Max for free tier, 3008 for Pro
    }
  },
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ]
}
```
**Warning:** Don't use both `builds` and `functions` together - choose one approach.

### Issue: Memory Errors
**Solution:** 
- Free tier limit: 2048MB (2GB)
- Pro tier limit: 3008MB (3GB)
- Ensure `ULTRALYTICS_NO_FASTSAM=1` is set (already configured)
- YOLO model loading is optimized to load once at startup
- Consider using Railway/Fly.io for better ML support if memory issues persist

### Issue: File Upload Fails
**Solution:**
- Files are saved to `/tmp` directory (Vercel's writable space)
- Files are automatically cleaned after function execution
- Consider uploading directly to Supabase Storage for persistence

### Issue: Cold Start Slow
**Solution:**
- YOLO model loads on first request (can be slow)
- Consider using Vercel Pro plan for better performance
- Or use Railway/Fly.io for persistent workers

### Issue: Session Not Persisting
**Solution:**
- Vercel uses HTTPS, so `SESSION_COOKIE_SECURE=True` is correct
- Check `FLASK_SECRET_KEY` is set
- Sessions work across serverless functions

---

## 📊 Performance Tips

1. **Use Vercel Pro Plan** for:
   - Longer timeouts (up to 300s)
   - Better cold start performance
   - More memory

2. **Optimize Model Loading:**
   - YOLO loads once per cold start
   - Consider caching model in `/tmp` (persists during warm period)

3. **File Storage:**
   - Use Supabase Storage for permanent file storage
   - `/tmp` is cleared after function execution

---

## 🔗 Useful Links

- [Vercel Python Documentation](https://vercel.com/docs/concepts/functions/serverless-functions/runtimes/python)
- [Vercel Environment Variables](https://vercel.com/docs/concepts/projects/environment-variables)
- [Vercel Pricing](https://vercel.com/pricing)

---

## ⚠️ Alternative Platforms

If you encounter issues with Vercel, consider:

1. **Railway** - Better for ML apps, easier setup
2. **Fly.io** - Good performance, global CDN
3. **Render** - Similar to current setup, already configured

See `DEPLOYMENT_GUIDE.md` for details.

---

## ✅ Post-Deployment Checklist

- [ ] Test login functionality
- [ ] Test image upload
- [ ] Verify detection works (may be slow on first request)
- [ ] Check session persistence
- [ ] Monitor function logs in Vercel dashboard
- [ ] Verify environment variables are set
- [ ] Test all dashboard pages

---

## 🎯 Quick Test

After deployment, test:
1. Visit: `https://your-app.vercel.app`
2. Login with your credentials
3. Upload a test image
4. Check detection results

If detection is slow, it's normal on Vercel due to cold starts. Consider Railway for better ML model performance.

