# Production Demo Deployment (VPS Reverse Proxy + Local Backend)

This setup keeps the application logic on your local server and uses the VPS only as:
- public entrypoint (`oncoview.dmitriderevjanko.com`)
- TLS termination (Let's Encrypt)
- reverse proxy to a reverse SSH tunnel endpoint

## Architecture

```text
Browser
  -> https://oncoview.dmitriderevjanko.com
  -> VPS Nginx (65.108.52.117, 443)
  -> 127.0.0.1:18080 on VPS (reverse SSH tunnel endpoint)
  -> local server (through SSH reverse tunnel)
  -> 127.0.0.1:18005 (uvicorn app.main:app)
```

## Port plan (safe defaults)

Use these dedicated ports to avoid conflicts with your other projects:
- Local API: `18005`
- VPS tunnel endpoint (loopback only): `18080`

Before applying, check both servers:

```bash
ss -tulpen | rg "18005|18080|8005|2223"
```

If any port is busy, pick another free high port pair and update all templates accordingly.

## 1) Local server setup (app service)

1. Copy service template:
   - `deploy/local/systemd/oncoview-api.service`
2. Edit `User=`, `Group=`, `WorkingDirectory=`, and model config paths if needed.
3. Install service:

```bash
sudo cp deploy/local/systemd/oncoview-api.service /etc/systemd/system/oncoview-api.service
sudo systemctl daemon-reload
sudo systemctl enable --now oncoview-api
sudo systemctl status oncoview-api --no-pager
curl -sS http://127.0.0.1:18005/health
```

## 2) Local server setup (persistent reverse tunnel)

1. Install `autossh`:

```bash
sudo apt-get update
sudo apt-get install -y autossh
```

2. Configure SSH key-based login from local server to VPS.
3. Copy and enable tunnel service:

```bash
sudo cp deploy/local/systemd/oncoview-reverse-tunnel.service /etc/systemd/system/oncoview-reverse-tunnel.service
sudo systemctl daemon-reload
sudo systemctl enable --now oncoview-reverse-tunnel
sudo systemctl status oncoview-reverse-tunnel --no-pager
```

4. Validate from VPS:

```bash
curl -sS http://127.0.0.1:18080/health
```

If this works, the tunnel is good.

## 3) VPS setup (Nginx + TLS)

1. Copy Nginx site config:
   - `deploy/vps/nginx/oncoview.conf`
2. Enable site:

```bash
sudo cp deploy/vps/nginx/oncoview.conf /etc/nginx/sites-available/oncoview.conf
sudo ln -s /etc/nginx/sites-available/oncoview.conf /etc/nginx/sites-enabled/oncoview.conf
sudo nginx -t
sudo systemctl reload nginx
```

3. Issue certificate:

```bash
sudo apt-get update
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d oncoview.dmitriderevjanko.com
sudo certbot renew --dry-run
```

## 4) GitHub push -> auto deploy

This repository includes:
- local deploy script: `deploy/local/deploy_pull_restart.sh`
- GitHub Action: `.github/workflows/deploy-through-tunnel.yml`

Flow:
1. Push to `main`.
2. GitHub Action SSHes to your local server through `65.108.52.117:2223`.
3. Runs deploy script (`git pull`, install deps if needed, restart service, health check).

### Required GitHub secrets

In repo settings -> Secrets and variables -> Actions, add:
- `LOCAL_DEPLOY_SSH_KEY` (private key that can login to local server user via tunnel)
- `LOCAL_DEPLOY_HOST` (set to `65.108.52.117`)
- `LOCAL_DEPLOY_PORT` (set to `2223`)
- `LOCAL_DEPLOY_USER` (for example `dmitri`)
- `LOCAL_DEPLOY_PATH` (for example `/home/dmitri/cancer_detector`)
- `LOCAL_DEPLOY_KNOWN_HOSTS` (output of `ssh-keyscan -p 2223 65.108.52.117`)

## 5) Zero-interference policy

To avoid affecting other projects:
- never bind new services to `0.0.0.0` unless required
- keep local API on `127.0.0.1`
- keep VPS tunnel endpoint on `127.0.0.1`
- only expose `80/443` publicly via Nginx
- use unique service names: `oncoview-*`

## Useful checks

```bash
# local
sudo systemctl status oncoview-api oncoview-reverse-tunnel --no-pager
journalctl -u oncoview-api -n 100 --no-pager
journalctl -u oncoview-reverse-tunnel -n 100 --no-pager

# vps
sudo nginx -t
sudo systemctl status nginx --no-pager
curl -I https://oncoview.dmitriderevjanko.com
```

