# Fly.io Deployment Script
# Run this script after adding payment info at: https://fly.io/dashboard/billing

Write-Host "=== Fly.io Deployment Script ===" -ForegroundColor Green
Write-Host ""

# Add Fly CLI to PATH
$env:PATH += ";C:\Users\Admin\.fly\bin"

# Check if Fly CLI is available
try {
    $flyVersion = flyctl --version
    Write-Host "✓ Fly CLI found: $flyVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Fly CLI not found. Please install it first." -ForegroundColor Red
    exit 1
}

# Step 1: Check login status
Write-Host "`n[1/6] Checking login status..." -ForegroundColor Cyan
try {
    $whoami = flyctl auth whoami
    Write-Host "✓ Logged in as: $whoami" -ForegroundColor Green
} catch {
    Write-Host "✗ Not logged in. Please run: flyctl auth login" -ForegroundColor Red
    exit 1
}

# Step 2: Create app
Write-Host "`n[2/6] Creating Fly.io app..." -ForegroundColor Cyan
try {
    flyctl apps create detectpechay --org personal
    Write-Host "✓ App created successfully" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to create app. Error: $_" -ForegroundColor Red
    Write-Host "  Make sure you've added payment info at: https://fly.io/dashboard/billing" -ForegroundColor Yellow
    exit 1
}

# Step 3: Set environment variables (secrets)
Write-Host "`n[3/6] Setting environment variables..." -ForegroundColor Cyan

# Read secrets from user or use defaults
$FLASK_SECRET_KEY = Read-Host "Enter FLASK_SECRET_KEY (or press Enter for default)"
if ([string]::IsNullOrWhiteSpace($FLASK_SECRET_KEY)) {
    $FLASK_SECRET_KEY = "change-this-secret-key-in-production-$(Get-Random)"
    Write-Host "  Using generated secret key" -ForegroundColor Yellow
}

$SUPABASE_URL = Read-Host "Enter SUPABASE_URL (or press Enter for default)"
if ([string]::IsNullOrWhiteSpace($SUPABASE_URL)) {
    $SUPABASE_URL = "https://zqkqmjlepigpwfykwzey.supabase.co"
}

$SUPABASE_ANON_KEY = Read-Host "Enter SUPABASE_ANON_KEY (or press Enter for default)"
if ([string]::IsNullOrWhiteSpace($SUPABASE_ANON_KEY)) {
    $SUPABASE_ANON_KEY = "sb_publishable_HNgog4XZVoR6FqaKuzIcGQ_7yrDAjFn"
}

try {
    flyctl secrets set `
        FLASK_SECRET_KEY="$FLASK_SECRET_KEY" `
        SUPABASE_URL="$SUPABASE_URL" `
        SUPABASE_ANON_KEY="$SUPABASE_ANON_KEY" `
        FLASK_ENV="production" `
        ULTRALYTICS_NO_FASTSAM="1"
    Write-Host "✓ Secrets set successfully" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to set secrets. Error: $_" -ForegroundColor Red
    exit 1
}

# Step 4: Deploy
Write-Host "`n[4/6] Deploying application..." -ForegroundColor Cyan
Write-Host "  This may take 5-10 minutes..." -ForegroundColor Yellow
try {
    flyctl deploy
    Write-Host "✓ Deployment completed successfully" -ForegroundColor Green
} catch {
    Write-Host "✗ Deployment failed. Error: $_" -ForegroundColor Red
    Write-Host "  Check logs with: flyctl logs" -ForegroundColor Yellow
    exit 1
}

# Step 5: Check status
Write-Host "`n[5/6] Checking app status..." -ForegroundColor Cyan
try {
    flyctl status
    Write-Host "✓ App is running" -ForegroundColor Green
} catch {
    Write-Host "⚠ Could not check status" -ForegroundColor Yellow
}

# Step 6: Open app
Write-Host "`n[6/6] Opening app in browser..." -ForegroundColor Cyan
try {
    flyctl open
    Write-Host "✓ App opened in browser" -ForegroundColor Green
} catch {
    Write-Host "⚠ Could not open browser automatically" -ForegroundColor Yellow
    Write-Host "  Visit: https://detectpechay.fly.dev" -ForegroundColor Cyan
}

Write-Host "`n=== Deployment Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Your app is live at: https://detectpechay.fly.dev" -ForegroundColor Cyan
Write-Host ""
Write-Host "Useful commands:" -ForegroundColor Yellow
Write-Host "  flyctl logs          - View logs" -ForegroundColor White
Write-Host "  flyctl status        - Check status" -ForegroundColor White
Write-Host "  flyctl open          - Open app" -ForegroundColor White
Write-Host "  flyctl secrets list  - View secrets" -ForegroundColor White

