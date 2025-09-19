[CmdletBinding()]
param(
    [string]$OutDir = $env:OUT,
    [string]$SpansPath,
    [string]$ScoreboardPath,
    [string]$OutputPath
)

function TryParse-Double([string]$s, [ref]$outVal) {
  $ci  = [System.Globalization.CultureInfo]::InvariantCulture
  $ns  = [System.Globalization.NumberStyles]::Float
  [double]$tmp = 0
  $text = if ($null -eq $s) { "" } else { $s.ToString() }
  $ok = [double]::TryParse($text, $ns, $ci, [ref]$tmp)
  if ($ok) { $outVal.Value = $tmp }
  return $ok
}

$ci = [System.Globalization.CultureInfo]::InvariantCulture

if (-not $OutDir) {
  throw "OUT directory not provided. Pass -OutDir or set the OUT environment variable."
}
$resolvedOut = Resolve-Path -Path $OutDir -ErrorAction Stop

if (-not $SpansPath) {
  $SpansPath = Join-Path $resolvedOut "highlight_spans.json"
}
if (-not (Test-Path $SpansPath)) {
  throw "Spans file not found: $SpansPath"
}

if (-not $ScoreboardPath) {
  $ScoreboardPath = Join-Path $resolvedOut "scoreboard_spans.json"
}
if (-not (Test-Path $ScoreboardPath)) {
  throw "Scoreboard file not found: $ScoreboardPath"
}

if (-not $OutputPath) {
  $OutputPath = Join-Path $resolvedOut "highlight_reel_plan.json"
}

$spanJson = Get-Content -Path $SpansPath -Raw -Encoding UTF8
$spanItems = @()
try {
  $spanData = $spanJson | ConvertFrom-Json
} catch {
  throw "Unable to parse spans JSON from $SpansPath"
}
foreach ($s in $spanData) {
  [double]$t0 = 0
  [double]$t1 = 0
  if (TryParse-Double $s.t0 ([ref]$t0) -and TryParse-Double $s.t1 ([ref]$t1)) {
    $duration = [math]::Max(0.0, $t1 - $t0)
    $segment = [pscustomobject]@{
      Start    = $t0
      End      = $t1
      Duration = $duration
      Label    = $s.label
    }
    $spanItems += $segment
  }
}

if ($ScoreboardPath.ToLower().EndsWith('.json')) {
  $scoreRaw = Get-Content -Path $ScoreboardPath -Raw -Encoding UTF8
  try {
    $scoreData = $scoreRaw | ConvertFrom-Json
  } catch {
    throw "Unable to parse scoreboard JSON from $ScoreboardPath"
  }
} else {
  $scoreData = Import-Csv -Path $ScoreboardPath
}

$scoreItems = @()
foreach ($r in $scoreData) {
  [double]$start = 0
  [double]$end = 0
  [double]$scoreValue = 0
  $startParsed = TryParse-Double $r.start ([ref]$start)
  $endParsed = TryParse-Double $r.end ([ref]$end)
  $scoreParsed = TryParse-Double $r.score ([ref]$scoreValue)
  $scoreText = if ($scoreParsed) { $scoreValue.ToString($ci) } else { $r.score }
  $item = [pscustomobject]@{
    Start = if ($startParsed) { $start } else { $null }
    End   = if ($endParsed) { $end } else { $null }
    Score = $scoreText
    Note  = $r.note
  }
  $scoreItems += $item
}

$plan = @{
  segments  = @()
  scoreboard = @()
}
foreach ($seg in $spanItems) {
  $plan.segments += @{
    start    = $seg.Start.ToString($ci)
    end      = $seg.End.ToString($ci)
    duration = $seg.Duration.ToString($ci)
    label    = $seg.Label
  }
}
foreach ($row in $scoreItems) {
  $plan.scoreboard += @{
    start = if ($null -ne $row.Start) { $row.Start.ToString($ci) } else { $null }
    end   = if ($null -ne $row.End) { $row.End.ToString($ci) } else { $null }
    score = $row.Score
    note  = $row.Note
  }
}

$filterParts = @()
for ($i = 0; $i -lt $spanItems.Count; $i++) {
  $seg = $spanItems[$i]
  $startStr = $seg.Start.ToString($ci)
  $endStr = $seg.End.ToString($ci)
  $filterParts += "[0:v]trim=start=$startStr:end=$endStr,setpts=PTS-STARTPTS[v$i];"
}
$plan.filter_complex = ($filterParts -join '')

$plan | ConvertTo-Json -Depth 5 | Set-Content -Path $OutputPath -Encoding UTF8

Write-Host ("Segments parsed: {0}" -f $spanItems.Count)
Write-Host ("Scoreboard entries: {0}" -f $scoreItems.Count)
Write-Host ("Plan saved to {0}" -f $OutputPath)
