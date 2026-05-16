# `aws/` — EC2 training toolkit

Persistent-instance training toolkit for the NFP Predictor. Master data lives in S3, the working copy lives on the EC2 instance's EBS volume, and you fire-and-forget training over SSH. The full `--train-all` walk-forward (NSA branch + post-training Kalman fusion) runs end-to-end in ~3–6 hours on the default `m7i.4xlarge` (16 vCPU, 64 GB RAM) — comfortable for a single overnight cycle.

The toolkit is **stateful but idempotent.** Re-running any script after a failure picks up where it left off. The EBS volume preserves the working copy between sessions so you typically only pay for compute while training is actively running.

## What gets provisioned

| Resource | Default name | Purpose |
|---|---|---|
| S3 bucket | `nfp-predictor-<account-id>` | Master data + outputs (encrypted at rest, versioned, public-blocked) |
| IAM role | `nfp-predictor-ec2-role` | Lets the instance read/write the bucket |
| Instance profile | `nfp-predictor-ec2-profile` | Attaches the role to the instance |
| Key pair | `nfp-predictor-key` | SSH key (private saved to `aws/.keys/`) |
| Security group | `nfp-predictor-sg` | Allows SSH from your current IP only (port 22) |
| EC2 instance | `nfp-predictor` (Name tag) | The training box (`m7i.4xlarge`, 250 GB gp3 root EBS) |

All defaults are in [`aws/config.env`](config.env). Override anything in `aws/config.local.env` (gitignored) — e.g. swap to `m7i.8xlarge` for memory-heavy Optuna sweeps, or set `INSTANCE_TYPE`, `ROOT_VOLUME_GB`, `S3_BUCKET`, `SSH_INGRESS_CIDR`.

## Files in `aws/`

### Local (run from your laptop)

| Script | What it does |
|---|---|
| [`provision.sh`](provision.sh) | One-time: creates bucket, IAM role + profile, key pair, security group, EC2 instance. Idempotent — re-run safely after partial failures. |
| [`teardown.sh`](teardown.sh) | One-time: tears everything down (with confirmation). |
| [`start.sh`](start.sh) | Starts the stopped instance, prints its public IP. |
| [`stop.sh`](stop.sh) | Stops the instance (EBS preserved, compute paused). |
| [`status.sh`](status.sh) | Shows instance state + public IP. |
| [`ssh.sh`](ssh.sh) | SSH to the running instance with the right key/options. |
| [`push_data.sh`](push_data.sh) | `aws s3 sync` ./data → s3. Training-only whitelist by default; `PUSH_DATA_ALL=1` syncs everything (~69 GB). |
| [`push_env.sh`](push_env.sh) | `scp` `.env` to the instance (never committed). |
| [`push_code.sh`](push_code.sh) | `rsync` code dirs to the instance. Excludes `data/`, `_output/`, `_temp/`, `continuous_futures/`, `economist_panel/`, caches, notebooks, `.env`. |
| [`pull_outputs.sh`](pull_outputs.sh) | `aws s3 sync` `s3://.../_output` → `./_output` after training. |
| [`run_training.sh`](run_training.sh) | Fire-and-forget: pushes latest code, launches detached `tmux` session, returns immediately. Auto-stops the instance when training finishes (unless `NO_AUTO_STOP=1`). |
| [`tail_log.sh`](tail_log.sh) | Live tail of the most recent training log on the instance. |

### On-instance (run after `ssh.sh`)

| Script | What it does |
|---|---|
| [`on_instance/bootstrap.sh`](on_instance/bootstrap.sh) | Idempotent: installs Python 3.12, OpenMP (`libomp-dev`) for LightGBM, AWS CLI v2, builds `.venv/`, installs all pip deps (lightgbm, shap, optuna, fredapi, yfinance, statsmodels, catboost, …). Auto-activates the venv on login. |
| [`on_instance/pull_data.sh`](on_instance/pull_data.sh) | `aws s3 sync s3://… → ~/NFP_Predictor/data`. |
| [`on_instance/run_training.sh`](on_instance/run_training.sh) | Wrapper invoked inside `tmux`: activates the venv, runs `python Train/train_lightgbm_nfp.py --train-all`, pushes `_output/` to S3, then `sudo shutdown -h +1` (unless `NO_AUTO_STOP=1`). All output `tee`'d to `_output/runs/run_<ts>.log`. |
| [`on_instance/push_outputs.sh`](on_instance/push_outputs.sh) | `aws s3 sync ./_output → s3://.../_output`. Always runs after training, even on failure. |

Everything is wired together by [`lib.sh`](lib.sh) — shared helpers for `aws` CLI invocation, instance metadata, the SSH options array, and an rsync wrapper that handles paths-with-spaces correctly (the project root literally lives under `~/Desktop/Github Repos/NFP_Predictor` on macOS).

## First-run checklist

```bash
# --- on your laptop --------------------------------------------------------
# 0. Install prerequisites
#    - AWS CLI v2 (`brew install awscli`)
#    - jq        (`brew install jq`)
#    - Configure an AWS profile named "nfp-predictor" (or set AWS_PROFILE)
#         aws configure --profile nfp-predictor

# 1. Provision (creates bucket, IAM, key pair, SG, EC2 instance)
./aws/provision.sh

# 2. Upload data to S3 (slow: ~50 GB whitelist, or 69 GB with PUSH_DATA_ALL=1)
./aws/push_data.sh
#   PUSH_DATA_ALL=1 ./aws/push_data.sh    # if you also want raw ETL inputs

# 3. Copy .env to the instance
./aws/push_env.sh

# 4. Push code
./aws/push_code.sh

# 5. SSH in and bootstrap the runtime
./aws/ssh.sh
ubuntu@ec2:~$ cd ~/NFP_Predictor
ubuntu@ec2:~/NFP_Predictor$ bash aws/on_instance/bootstrap.sh
ubuntu@ec2:~/NFP_Predictor$ bash aws/on_instance/pull_data.sh
ubuntu@ec2:~/NFP_Predictor$ exit
```

## Day-to-day usage

```bash
# Standard cycle: edit code locally, fire-and-forget on AWS.

./aws/run_training.sh                            # default: --train-all
./aws/run_training.sh --iterate-fusion-tune       # joint NSA ↔ Kalman tune
NO_AUTO_STOP=1 ./aws/run_training.sh             # leave instance up after training

./aws/tail_log.sh                                 # watch live log (Ctrl-C to detach)
./aws/ssh.sh                                      # if you want to attach to the tmux session:
                                                  #   tmux a -t train

# When the instance has stopped itself (auto-shutdown ~1 min after training),
# pull outputs back to your laptop:
./aws/pull_outputs.sh
```

What `run_training.sh` does, in order:

1. Starts the instance if stopped (`start.sh`).
2. Waits for SSH to come up.
3. `rsync`s your latest code (`push_code.sh`).
4. Launches a **detached** `tmux` session called `train` running [`on_instance/run_training.sh`](on_instance/run_training.sh) with whatever args you passed (default `--train-all`).
5. Returns immediately — close your laptop, the training keeps going.

The on-instance wrapper then runs the training, syncs `_output/` to S3 (even if training fails), and schedules `sudo shutdown -h +1`. Because `InstanceInitiatedShutdownBehavior` defaults to `stop` (not `terminate`), the EBS volume is preserved.

## Operational notes

- **Cost shape.** You pay EBS storage (~$25/mo for 250 GB gp3) continuously; compute (~$0.81/hr for `m7i.4xlarge` on-demand in us-east-1) only when running. Auto-stop is the default for exactly this reason.
- **Key rotation.** The private key is saved once to `aws/.keys/<KEY_NAME>.pem` (mode 600). Back it up to your password manager — `provision.sh` will refuse to re-create it if the AWS-side key pair exists but the local file doesn't, to avoid an unrecoverable mismatch.
- **SSH ingress.** `SSH_INGRESS_CIDR=auto` (the default) detects your current public IP via `checkip.amazonaws.com` and locks the SG to `<your-ip>/32`. If you change networks, re-run `provision.sh` — it will idempotently authorize the new CIDR.
- **`push_code.sh` deliberately excludes `.env`.** Use `push_env.sh` instead so you don't accidentally clobber the instance's `.env` with a local one (or commit it to S3 via a stray sync).
- **Data sync whitelist.** `push_data.sh` ships the *training-only* subset by default: `master_snapshots/`, `fred_data/`, `NFP_target/`. The full ETL inputs (`Exogenous_data/`, `fred_data_prepared_{nsa,sa}/`) are only needed if you plan to re-run the data pipeline on AWS — set `PUSH_DATA_ALL=1` to include them.
- **Re-running cleanly.** `provision.sh`, `bootstrap.sh`, and `push_data.sh` are all idempotent. Failures mid-flow are recoverable without state surgery.

## Configuration (`config.env`)

| Variable | Default | Notes |
|---|---|---|
| `AWS_PROFILE` | `nfp-predictor` | AWS CLI profile |
| `AWS_REGION` | `us-east-1` | |
| `PROJECT_NAME` | `nfp-predictor` | Drives every resource name |
| `INSTANCE_TYPE` | `m7i.4xlarge` | 16 vCPU, 64 GB. Step up to `m7i.8xlarge` for memory-heavy Optuna sweeps |
| `UBUNTU_AMI_SSM_PARAM` | Ubuntu 24.04 LTS AMI from SSM | Never hardcode an AMI — they rotate |
| `ROOT_VOLUME_GB` | `250` | Data is ~69 GB; headroom for outputs + Optuna study DBs + venv |
| `ROOT_VOLUME_TYPE` | `gp3` | |
| `S3_BUCKET` | derived: `${PROJECT_NAME}-<account-id>` | Override in `config.local.env` for custom name |
| `S3_DATA_PREFIX` | `data` | |
| `S3_OUTPUT_PREFIX` | `_output` | |
| `REMOTE_USER` | `ubuntu` | |
| `REMOTE_PROJECT_DIR` | `/home/ubuntu/NFP_Predictor` | |
| `SSH_INGRESS_CIDR` | `auto` | Or set to a specific CIDR like `203.0.113.10/32` |

Override anything in `aws/config.local.env` (gitignored). See [`config.local.env.example`](config.local.env.example) for the recommended template.

## Teardown

```bash
./aws/teardown.sh
```

Asks for confirmation, then terminates the instance, deletes the key pair, removes the security group, detaches/deletes the instance profile + role, and (optionally) deletes the S3 bucket. The `.state/` and `.keys/` directories are kept locally so you have an audit trail; delete them manually if you want a fully clean slate.

## Related

- The same flow is also driven from a laptop without AWS via the local entrypoints — see [`../README.md`](../README.md) §12.
- IAM policies live in [`iam/`](iam/) — trust policy, instance policy (templated with `__BUCKET__`), and a CLI-user policy for reference.
