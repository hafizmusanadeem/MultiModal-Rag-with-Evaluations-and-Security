# deploy.ps1 — one-shot Cloud Run deploy for the RAG backend (Windows PowerShell).
#
# Run from the PROJECT ROOT (the folder containing app/ and Dockerfile):
#   .\deploy\deploy.ps1 -Project YOUR_GCP_PROJECT_ID
#
# Prereqs: gcloud installed, `gcloud auth login` done, billing enabled on the project.
# What it does:
#   1. enables the needed Google APIs
#   2. stores your .env keys in Secret Manager (+ generates the backend API key on first run)
#   3. grants the Cloud Run service account access to those secrets
#   4. builds the image from ./Dockerfile and deploys it
# Re-running is safe: secrets get a new version, the API key is NOT rotated.

param(
    [Parameter(Mandatory = $true)][string]$Project,
    [string]$Region = "us-central1",
    [string]$Service = "rag-api",
    [int]$MinInstances = 0,           # use 1 before sharing the link (no cold start; costs a few $/month)
    [string]$EnvFile = ".env"
)

# Native gcloud writes progress to stderr; don't let PowerShell treat that as a failure.
$ErrorActionPreference = "Continue"

function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

if (-not (Test-Path "Dockerfile")) { Fail "Run this from the project root (Dockerfile not found here)." }
if (-not (Test-Path $EnvFile))     { Fail "$EnvFile not found." }
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) { Fail "gcloud CLI not found on PATH." }

# ---- 1. APIs ---------------------------------------------------------------
Write-Host "Enabling APIs..." -ForegroundColor Cyan
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com --project $Project
if ($LASTEXITCODE -ne 0) { Fail "Could not enable APIs (is billing enabled on '$Project'?)." }

# ---- 2. Secrets ------------------------------------------------------------
# Parse .env  (KEY = "value" or KEY=value; comment lines are ignored)
$vals = @{}
Get-Content $EnvFile | ForEach-Object {
    if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
        $vals[$Matches[1]] = $Matches[2].Trim('"').Trim("'")
    }
}

function Set-Secret([string]$Name, [string]$Value) {
    $tmp = [IO.Path]::GetTempFileName()
    try {
        [IO.File]::WriteAllText($tmp, $Value)      # no trailing newline
        gcloud secrets describe $Name --project $Project *> $null
        if ($LASTEXITCODE -eq 0) {
            gcloud secrets versions add $Name --data-file=$tmp --project $Project *> $null
        } else {
            gcloud secrets create $Name --data-file=$tmp --replication-policy=automatic --project $Project *> $null
        }
        if ($LASTEXITCODE -ne 0) { Fail "Could not write secret $Name." }
    } finally { Remove-Item $tmp -ErrorAction SilentlyContinue }
}

$required = @("GROQ_API_KEY", "PORTKEY_API_KEY", "PORTKEY_CONFIG_ID",
              "QDRANT_API_KEY", "QDRANT_CLUSTER_ENDPOINT", "GEMINI_API_KEY")
$optional = @("GROQ_FALLBACK_API_KEY", "LOGFIRE_TOKEN", "LANGSMITH_API_KEY")

$mappings = @()
foreach ($k in $required + $optional) {
    $v = $vals[$k]
    if ([string]::IsNullOrWhiteSpace($v)) {
        if ($required -contains $k) { Fail "$k is empty in $EnvFile (required)." }
        Write-Host "  skipping optional $k (empty)" -ForegroundColor Yellow
        continue
    }
    $secretName = "rag-" + $k.ToLower().Replace("_", "-")
    Set-Secret $secretName $v
    $mappings += "$k=${secretName}:latest"
    Write-Host "  secret ok: $secretName"
}

# Backend API key: generated once, never rotated by re-runs.
gcloud secrets describe rag-api-key --project $Project *> $null
if ($LASTEXITCODE -ne 0) {
    $bytes = New-Object byte[] 32
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $apiKey = -join ($bytes | ForEach-Object { $_.ToString("x2") })
    Set-Secret "rag-api-key" $apiKey
    Write-Host ""
    Write-Host "  GENERATED BACKEND API KEY (save it — it goes into the Streamlit secrets):" -ForegroundColor Green
    Write-Host "  $apiKey" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host "  rag-api-key already exists — kept (read it with: gcloud secrets versions access latest --secret rag-api-key --project $Project)"
}
$mappings += "API_KEY=rag-api-key:latest"

# ---- 3. IAM for the default compute service account ------------------------
$projectNumber = (gcloud projects describe $Project --format="value(projectNumber)").Trim()
$sa = "serviceAccount:$projectNumber-compute@developer.gserviceaccount.com"
Write-Host "Granting roles to $sa ..." -ForegroundColor Cyan
foreach ($role in @("roles/secretmanager.secretAccessor", "roles/cloudbuild.builds.builder")) {
    gcloud projects add-iam-policy-binding $Project --member=$sa --role=$role --condition=None *> $null
}

# ---- 4. Build + deploy ------------------------------------------------------
$tracing = if ($vals["LANGSMITH_API_KEY"]) { "true" } else { "false" }
$project_ls = if ($vals["LANGSMITH_PROJECT"]) { $vals["LANGSMITH_PROJECT"] } else { "rag_scale_test" }
$envVars = "REQUIRE_GEMINI=1,LANGSMITH_TRACING=$tracing,LANGSMITH_PROJECT=$project_ls"

Write-Host "Building and deploying (first build takes several minutes)..." -ForegroundColor Cyan
gcloud run deploy $Service `
    --project $Project --region $Region `
    --source . `
    --allow-unauthenticated `
    --memory 2Gi --cpu 2 --cpu-boost `
    --min-instances $MinInstances --max-instances 1 `
    --timeout 300 `
    --set-env-vars $envVars `
    --set-secrets ($mappings -join ",")
if ($LASTEXITCODE -ne 0) { Fail "Deploy failed. Read the build log URL printed above; see DEPLOY.md > Troubleshooting." }

$url = (gcloud run services describe $Service --project $Project --region $Region --format="value(status.url)").Trim()
Write-Host ""
Write-Host "Deployed: $url" -ForegroundColor Green
Write-Host "Smoke test:"
Write-Host "  curl.exe $url/health"
Write-Host "  curl.exe -X POST $url/query -H `"Content-Type: application/json`" -H `"X-API-Key: <your key>`" -d '{`"q`":`"What is an FZE?`"}'"
