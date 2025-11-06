# Railway Persistent Storage Setup

## Problem
AI logs and database are currently ephemeral - they get deleted on restart!

## Solution: Add Railway Volume

### Step 1: Create Volume
1. Go to Railway dashboard
2. Click on your backend service
3. Go to **"Variables"** tab
4. Scroll down to **"Volumes"** section
5. Click **"New Volume"**
6. **Mount Path**: `/data`
7. Click **"Add"**

### Step 2: Update Environment Variable
In Railway **"Variables"** tab, add:
```
DB_DIR=/data
LOG_DIR=/data/logs
```

### Step 3: Redeploy
Click **"Redeploy"** button

---

## What Gets Saved
- ✅ **Database**: `/data/trades.db` (all trades, balance, settings)
- ✅ **AI Logs**: `/data/logs/llm_reasoning_YYYYMMDD.jsonl`
- ✅ **Trade CSV**: `/data/logs/trades_YYYYMMDD.csv`

## Volume Size
- Default: 1GB (plenty for logs and database)
- Can increase if needed

---

## Verify It Works
After redeploying, check Railway logs for:
```
✅ Database tables created/verified
✅ Log directory created: /data/logs
```

