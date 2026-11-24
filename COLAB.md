# Run the pipeline on Google Colab (Colab Pro) from your terminal

Uses Google's official Colab CLI. Your terminal (WSL Ubuntu) starts a Colab GPU, sends the code,
runs the pipeline there and downloads the results. Claude Code can drive all of it.
The Colab CLI supports Linux and macOS only, so use WSL, not PowerShell.

## One-time setup (WSL Ubuntu)

```bash
# 1. Colab CLI
curl -LsSf https://astral.sh/uv/install.sh | sh      # skip if uv already works
uv tool install google-colab-cli
colab version

# 2. Log in with the Google account that has Colab Pro
colab sessions
#    It prints a URL: open it in the browser, choose your Colab Pro account,
#    copy the code it shows, paste it back into the terminal.

# 3. Give Claude Code Google's Colab skill (teaches it the colab commands)
mkdir -p ~/.claude/skills/colab-operator
colab skill > ~/.claude/skills/colab-operator/SKILL.md
```

## Run

From the repo folder:
```bash
GPU=A100 colab/run_on_colab.sh both      # or: bowler / batsman
```
- Starts an A100 session called `cricket`, clones `Pranahv-main` from GitHub, adds your local code changes, runs both clips, prints progress every minute.
- Results: `colab_results/runs/colab_bowler/` and `colab_results/runs/colab_batsman/`.
- The VM is stopped at the end. If you press Ctrl+C it is stopped too. Check with `colab sessions`.
- Cost: an A100 uses about 15 compute units per hour (Colab Pro gives 100 a month). `GPU=L4` or `GPU=T4` is cheaper. `colab usage` shows your balance.

Optional, to keep the 6 GB of models between sessions: start the session yourself first and mount Drive.
```bash
colab new -s cricket --gpu A100
colab drivemount -s cricket          # follow the browser prompt
GPU=A100 colab/run_on_colab.sh both  # reuses the session and caches models on Drive
```

## With Claude Code

```bash
cd <repo folder>
claude
```
Then ask, for example:
> Run the full pipeline on Colab with an A100 using colab/run_on_colab.sh, watch it, and when it's done show me the PCK scores and where the final videos are.

`CLAUDE.md` in the repo tells Claude Code how the project and the Colab runner work.

## If something fails
| Symptom | Fix |
|---|---|
| `colab: command not found` | Close and reopen Ubuntu, or `export PATH="$HOME/.local/bin:$PATH"` |
| 401 / 403 from any `colab` command | Log in again: `colab sessions` |
| `colab new` fails with 400 for A100 | No A100 available right now: `GPU=L4` or `GPU=T4` |
| Log ends with `PIPELINE_DONE rc=3` | The VM has no GPU: you got a CPU runtime |
| Log ends with `rc=5` | Generation failed: read `colab_results/job.log` |
| Results missing after Ctrl+C | The VM was stopped; run again |
