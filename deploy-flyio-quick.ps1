# Quick Fly.io Deployment (uses default values)
# Run this after adding payment info at: https://fly.io/dashboard/billing

$env:PATH += ";C:\Users\Admin\.fly\bin"

Write-Host "=== Quick Fly.io Deployment ===" -ForegroundColor Green

# Create app
Write-Host "Creating app..." -ForegroundColor Cyan
flyctl apps create detectpechay --org personal

# Set secrets with default values
Write-Host "Setting secrets..." -ForegroundColor Cyan
flyctl secrets set `
    FLASK_SECRET_KEY="change-this-secret-key-$(Get-Random)" `
    SUPABASE_URL="https://zqkqmjlepigpwfykwzey.supabase.co" `
    SUPABASE_ANON_KEY="sb_publishable_HNgog4XZVoR6FqaKuzIcGQ_7yrDAjFn" `
    FLASK_ENV="production" `
    ULTRALYTICS_NO_FASTSAM="1"

# Deploy
Write-Host "Deploying..." -ForegroundColor Cyan
flyctl deploy

# Open app
Write-Host "Opening app..." -ForegroundColor Cyan
flyctl open

Write-Host "`n✓ Deployment complete!" -ForegroundColor Green

