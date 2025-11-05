# 🚀 Deployment Guide - FREE 24/7 Hosting

This guide shows you how to deploy your trading bot **completely FREE** using:
- **Vercel** for the frontend (React)
- **Fly.io** for the backend (FastAPI)

---

## 📋 Prerequisites

1. **GitHub account** (to push your code)
2. **Vercel account** (sign up at [vercel.com](https://vercel.com) - FREE)
3. **Fly.io account** (sign up at [fly.io](https://fly.io) - FREE tier)
4. **DeepSeek API key** (get from [platform.deepseek.com](https://platform.deepseek.com))

---

## 🎯 Part 1: Deploy Backend to Fly.io

### Step 1: Install Fly CLI

```bash
# Windows (PowerShell)
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"

# Mac/Linux
curl -L https://fly.io/install.sh | sh
```

### Step 2: Login to Fly.io

```bash
fly auth login
```

### Step 3: Launch Your App

```bash
cd backend
fly launch
```

When prompted:
- **App name**: `traderbot-api-YOUR_NAME` (must be unique)
- **Region**: Choose closest to you
- **PostgreSQL**: `No` (we use SQLite)
- **Redis**: `No`
- **Deploy now**: `No` (we need to set secrets first)

### Step 4: Create Persistent Storage

```bash
# Create a volume for database and logs
fly volumes create traderbot_data --size 1
```

### Step 5: Set Environment Variables (Secrets)

```bash
fly secrets set DEEPSEEK_API_KEY="your_actual_deepseek_api_key_here"
fly secrets set KRAKEN_API_KEY="your_kraken_api_key"
fly secrets set KRAKEN_SECRET="your_kraken_secret"
fly secrets set INITIAL_BALANCE=10000
fly secrets set RISK_PER_TRADE=0.02
fly secrets set MAX_TRADES=3
```

**Note**: Replace `your_actual_deepseek_api_key_here` with your real API key!

### Step 6: Deploy Backend

```bash
fly deploy
```

### Step 7: Get Your Backend URL

```bash
fly info
```

You'll see something like: `https://traderbot-api-YOUR_NAME.fly.dev`

**SAVE THIS URL** - you'll need it for the frontend!

---

## 🎨 Part 2: Deploy Frontend to Vercel

### Step 1: Push Code to GitHub

```bash
cd ..
git init
git add .
git commit -m "Initial commit - Trading Bot"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/traderbot.git
git push -u origin main
```

### Step 2: Import to Vercel

1. Go to [vercel.com](https://vercel.com)
2. Click **"Add New..."** → **"Project"**
3. Import your GitHub repository
4. Configure project:
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`

### Step 3: Add Environment Variable

Before deploying, add your backend URL:

1. In Vercel project settings → **Environment Variables**
2. Add:
   - **Key**: `VITE_API_URL`
   - **Value**: `https://traderbot-api-YOUR_NAME.fly.dev` (your Fly.io URL)
   - **Environment**: All (Production, Preview, Development)

### Step 4: Deploy

Click **"Deploy"** and wait ~2 minutes.

Your app will be live at: `https://traderbot.vercel.app`

---

## 🔧 Configuration

### Update Frontend API URL

If you need to change the backend URL later:

```bash
# In Vercel dashboard
vercel env add VITE_API_URL
# Enter your Fly.io URL: https://traderbot-api-YOUR_NAME.fly.dev
```

### Update Backend Environment Variables

```bash
cd backend
fly secrets set DEEPSEEK_API_KEY="new_key_here"
```

---

## 📊 Monitoring & Logs

### View Backend Logs (Fly.io)

```bash
cd backend
fly logs
```

### View App Status

```bash
fly status
```

### SSH into Backend (if needed)

```bash
fly ssh console
```

---

## 💰 Cost Breakdown

| Service | Free Tier | Your Usage | Cost |
|---------|-----------|------------|------|
| **Vercel** | Unlimited hobby sites | Frontend | **$0/month** |
| **Fly.io** | 3 VMs (256MB each) | 1 VM | **$0/month** |
| **DeepSeek API** | Pay-as-you-go | ~720 calls/month | **~$6-12/month** |
| **TOTAL** | | | **$6-12/month** |

---

## 🔄 Updating Your App

### Update Backend

```bash
cd backend
git pull origin main
fly deploy
```

### Update Frontend

Just push to GitHub - Vercel auto-deploys:

```bash
cd frontend
git add .
git commit -m "Update frontend"
git push origin main
```

---

## 🐛 Troubleshooting

### Backend won't start

```bash
fly logs  # Check error messages
fly secrets list  # Verify secrets are set
```

### Frontend can't connect to backend

1. Check CORS in `backend/main.py` - should allow your Vercel domain
2. Verify `VITE_API_URL` in Vercel environment variables
3. Check backend is running: visit `https://your-app.fly.dev/` in browser

### Database not persisting

```bash
fly volumes list  # Check volume exists
fly ssh console
cd /app/data
ls -la  # Check if trades.db exists
```

---

## 🛡️ Security Notes

1. **Never commit `.env` files** - they're in `.gitignore`
2. **Use Fly.io secrets** for sensitive data
3. **Use Vercel environment variables** for API URLs
4. **Enable HTTPS** (automatic on both platforms)

---

## 🎉 You're Live!

Your trading bot is now running 24/7 for FREE (except DeepSeek API costs)!

- **Frontend**: `https://traderbot.vercel.app`
- **Backend**: `https://traderbot-api-YOUR_NAME.fly.dev`
- **Auto-trading**: Running every hour automatically

---

## 📞 Need Help?

- Fly.io docs: https://fly.io/docs
- Vercel docs: https://vercel.com/docs
- Check logs: `fly logs` and Vercel dashboard
