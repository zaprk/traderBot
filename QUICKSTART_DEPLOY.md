# ⚡ Quick Deploy - 5 Minutes to Live

## 🎯 Fast Track (Copy-Paste Commands)

### 1️⃣ Deploy Backend (Fly.io)

```bash
# Install Fly CLI
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"

# Login
fly auth login

# Deploy backend
cd backend
fly launch --name traderbot-api-YOURNAME --region ord --no-deploy

# Create storage
fly volumes create traderbot_data --size 1

# Set secrets (REPLACE WITH YOUR KEYS!)
fly secrets set DEEPSEEK_API_KEY="sk-your-actual-key-here"
fly secrets set INITIAL_BALANCE=10000
fly secrets set RISK_PER_TRADE=0.02
fly secrets set MAX_TRADES=3

# Deploy!
fly deploy

# Get your URL
fly info
```

**✅ Copy your backend URL**: `https://traderbot-api-YOURNAME.fly.dev`

---

### 2️⃣ Deploy Frontend (Vercel)

```bash
# Push to GitHub
cd ..
git init
git add .
git commit -m "Deploy trading bot"
git branch -M main

# Create GitHub repo first at: https://github.com/new
# Then:
git remote add origin https://github.com/YOUR_USERNAME/traderbot.git
git push -u origin main
```

**Then:**
1. Go to [vercel.com](https://vercel.com)
2. Click **"Add New Project"**
3. Import your GitHub repo
4. Set **Root Directory**: `frontend`
5. Add environment variable:
   - **VITE_API_URL** = `https://traderbot-api-YOURNAME.fly.dev`
6. Click **Deploy**

---

## ✅ Done! Your app is LIVE 24/7

- **Frontend**: `https://traderbot.vercel.app`
- **Backend**: `https://traderbot-api-YOURNAME.fly.dev`
- **Cost**: $0 hosting + ~$10/month DeepSeek API

---

## 🔧 Quick Commands

```bash
# View logs
fly logs

# Update backend
cd backend && fly deploy

# Update frontend (just push to GitHub)
git add . && git commit -m "update" && git push
```

See `DEPLOYMENT.md` for full details!

