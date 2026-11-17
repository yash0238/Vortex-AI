#!/usr/bin/env pwsh
# DDP Launch Script for VORTEX-AI on 2x T4 GPUs (PowerShell)
# Usage: ./scripts/train_ddp.ps1 [-Config training/config.yaml] [-AdditionalArgs @()]

param(
    [string]$Config = "training/config.yaml",
    [string[]]$AdditionalArgs = @()
)

$ErrorActionPreference = "Stop"

# DDP environment
$env:MASTER_ADDR = "localhost"
$env:MASTER_PORT = "29500"
$env:NCCL_DEBUG = "INFO"
$env:NCCL_SOCKET_IFNAME = "lo"
$env:PYTHONFAULTHANDLER = "1"

$GPUS = 2

Write-Host "========================================="
Write-Host "VORTEX-AI DDP Training Launch"
Write-Host "Config: $Config"
Write-Host "GPUs: $GPUS"
Write-Host "Master: $env:MASTER_ADDR:$env:MASTER_PORT"
Write-Host "========================================="

# Launch with torchrun
$torchrunArgs = @(
    "--nproc_per_node=$GPUS",
    "--nnodes=1",
    "--master_addr=$env:MASTER_ADDR",
    "--master_port=$env:MASTER_PORT",
    "train_vortex.py",
    "--config", $Config
) + $AdditionalArgs

& torchrun @torchrunArgs