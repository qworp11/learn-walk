# Скачивание LAFAN1 Retargeting Dataset (G1) с HuggingFace
$files = @(
  "dance1_subject1.csv","dance1_subject2.csv","dance1_subject3.csv",
  "dance2_subject1.csv","dance2_subject2.csv","dance2_subject3.csv","dance2_subject4.csv","dance2_subject5.csv",
  "fallAndGetUp1_subject1.csv","fallAndGetUp1_subject4.csv","fallAndGetUp1_subject5.csv",
  "fallAndGetUp2_subject2.csv","fallAndGetUp2_subject3.csv","fallAndGetUp3_subject1.csv",
  "fight1_subject2.csv","fight1_subject3.csv","fight1_subject5.csv",
  "fightAndSports1_subject1.csv","fightAndSports1_subject4.csv",
  "jumps1_subject1.csv","jumps1_subject2.csv","jumps1_subject5.csv",
  "run1_subject2.csv","run1_subject5.csv","run2_subject1.csv","run2_subject4.csv",
  "sprint1_subject2.csv","sprint1_subject4.csv",
  "walk1_subject1.csv","walk1_subject2.csv","walk1_subject5.csv",
  "walk2_subject1.csv","walk2_subject3.csv","walk2_subject4.csv",
  "walk3_subject1.csv","walk3_subject2.csv","walk3_subject3.csv","walk3_subject4.csv","walk3_subject5.csv",
  "walk4_subject1.csv"
)
$base = "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/main/g1"
$dir = "G:\LEARN WALK\data\motions\lafan1_g1"
New-Item -ItemType Directory -Force $dir | Out-Null
foreach ($f in $files) {
  $out = Join-Path $dir $f
  if (Test-Path $out) { Write-Host "SKIP $f"; continue }
  curl.exe -sL --retry 3 -o $out "$base/$f?download=true"
  if ($LASTEXITCODE -eq 0) { Write-Host "OK $f" } else { Write-Host "FAIL $f" }
}
Write-Host "ALL_DONE"
