# Running eeANE as a launchd service

This guide sets up `eeane serve` as a macOS user LaunchAgent, so it
starts automatically when you log in and is restarted by launchd if it
ever crashes or is killed. It pairs well with eeANE's default on-demand
loading: the agent can stay running at all times while idle, since no
model is actually loaded onto the Neural Engine (and no memory is held
for it) until a request for it arrives.

## Prerequisites

- eeANE is installed and `eeane` runs — see the main
  [README.md](../README.md) for install instructions.
- Every model referenced by your config has already been compiled with
  `eeane compile <model>`.
- Your config file passes validation:

  ```sh
  eeane check-config --config /path/to/eeane.toml
  ```

- You know the absolute path to the `eeane` executable. Installation
  method determines where it lands (for example `uv tool install` and
  `pipx` both default to `~/.local/bin/eeane`, but this varies), so
  confirm it yourself:

  ```sh
  which eeane
  ```

## Install the agent

1. Copy the template into your per-user LaunchAgents directory:

   ```sh
   mkdir -p ~/Library/LaunchAgents
   cp docs/eeane.launchd.plist ~/Library/LaunchAgents/local.eeane.serve.plist
   ```

2. Edit `~/Library/LaunchAgents/local.eeane.serve.plist` and replace the
   three placeholder paths with real, absolute paths:
   - the `eeane` executable path (from `which eeane` above),
   - the `--config` path to your `eeane.toml`,
   - the `StandardOutPath` / `StandardErrorPath` log file (its parent
     directory must exist — for example `mkdir -p ~/Library/Logs/eeane`).

3. Load the agent:

   ```sh
   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/local.eeane.serve.plist
   ```

## Verify

Check that the server answers on the host/port from your config (the
default is `127.0.0.1:7997`; adjust if your config sets something else):

```sh
curl http://127.0.0.1:7997/health
```

You can also confirm the process is registered with launchd:

```sh
launchctl print gui/$(id -u)/local.eeane.serve
```

## Manage

Stop the service:

```sh
launchctl bootout gui/$(id -u)/local.eeane.serve
```

Restart it (for example after editing the config):

```sh
launchctl kickstart -k gui/$(id -u)/local.eeane.serve
```

Follow the logs:

```sh
tail -f ~/Library/Logs/eeane/eeane.log
```

## Updating eeANE

After upgrading the package (for example `uv tool upgrade eeane`),
restart the service so it picks up the new version:

```sh
launchctl kickstart -k gui/$(id -u)/local.eeane.serve
```

## Notes and limitations

- This is a user **LaunchAgent**, which only runs while its owner is
  logged into a GUI session. Running eeANE around the clock on a
  machine that is not always logged in requires additional setup (such
  as automatic login) that is outside the scope of this guide. Running
  eeANE as a root **LaunchDaemon** instead is not documented or
  verified here.
- Every path inside the plist must be an absolute path; launchd does
  not resolve a shell `PATH` or `~` for you.
- `EnvironmentVariables` is normally unnecessary in the plist: pass
  `--config` with an absolute path instead, and put any API key in the
  config file itself (see [README.md](../README.md)) rather than the
  environment.
