# Swap OpenSSH and container-rl-ssh ports

Current state:
- OpenSSH: port 22
- container-rl-ssh: port 2222

Goal:
- OpenSSH: port 2222 (admin access)
- container-rl-ssh: port 22 (users just type `ssh hostname`)

## Step 1 — Move OpenSSH to port 2222

```bash
sudo sed -i 's/^#Port 22/Port 2222/' /etc/ssh/sshd_config
sudo sed -i 's/^Port 22/Port 2222/' /etc/ssh/sshd_config
sudo systemctl restart sshd
```

Your current session stays alive because it uses `restart`, not `stop`.

## Step 2 — Verify OpenSSH still works

From a second terminal (keep your current one open!):

```bash
ssh -p 2222 root@72.62.132.174
```

If you get locked out, fix `/etc/ssh/sshd_config` from your existing session.

## Step 3 — Move container-rl-ssh to port 22

```bash
sudo systemctl stop container-rl-ssh
sudo sed -i 's/--addr :2222/--addr :22/' /etc/systemd/system/container-rl-ssh.service
sudo systemctl daemon-reload
sudo systemctl start container-rl-ssh
```

## Step 4 — Verify both work

```bash
# Container game — no -p needed (port 22)
ssh container-rl.example.com

# OpenSSH admin — needs -p 2222
ssh -p 2222 root@72.62.132.174
```

## Rolling back

If something breaks:

```bash
# Put OpenSSH back on 22
sudo sed -i 's/^Port 2222/Port 22/' /etc/ssh/sshd_config
sudo systemctl restart sshd

# Put container-rl-ssh back on 2222
sudo systemctl stop container-rl-ssh
sudo sed -i 's/--addr :22/--addr :2222/' /etc/systemd/system/container-rl-ssh.service
sudo systemctl daemon-reload
sudo systemctl start container-rl-ssh
```
