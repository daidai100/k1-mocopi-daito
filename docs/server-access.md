# Server access

Use `ssh server-wired` for the direct wired Ethernet connection, including
large artifact transfers and training status checks. Use `ssh server` for the
Tailscale connection. Both aliases select the same server.

Example: `rsync -a --partial bundle/ server-wired:/mnt/ssd1/k1-motion/experiments/NAME/bundle/`
