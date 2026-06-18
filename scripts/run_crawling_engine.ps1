$engineDir = "C:\Users\Lutfi\Documents\Project\AITF\crawler-search-engine\Windows"
$sourceKeywordDir = "C:\Users\Lutfi\Documents\Project\AITF\rutilahu-vlm-etl\data\keywords"
$outputRoot = "C:\Users\Lutfi\Documents\Project\AITF\rutilahu-vlm-etl\data\crawler_outputs"

$engineExe = Join-Path $engineDir "aitf-engine.exe"
$workingKeywordFile = Join-Path $engineDir "daftar_keyword.txt"
$engineOutputFile = Join-Path $engineDir "output\crawler_url_image.txt"

$token = "R8trcqTXSWyPPnkIbOI6QKV3dm+2xsGeP90Eijj8AR/dunl4VnogBbsa1YcJsH65cAdsfdG0wvJ2qTj7DSYiRA=="

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$keywordFiles = Get-ChildItem $sourceKeywordDir -Filter *.txt | Sort-Object Name

foreach ($file in $keywordFiles)
{
    Write-Host ""
    Write-Host "==================================="
    Write-Host "Processing: $($file.Name)"
    Write-Host "==================================="

    # Hapus output lama jika masih ada
    if (Test-Path $engineOutputFile)
    {
        Remove-Item $engineOutputFile -Force
    }

    # Copy keyword file ke daftar_keyword.txt
    Copy-Item $file.FullName $workingKeywordFile -Force

    Push-Location $engineDir

    try
    {
        & $engineExe `
            crawl `
            --search-image-mode=True `
            --token=$token

        $exitCode = $LASTEXITCODE

        if ($exitCode -ne 0)
        {
            Write-Warning "Crawler exited with code $exitCode"
        }
    }
    finally
    {
        Pop-Location
    }

    if (Test-Path $engineOutputFile)
    {
        $safeName = $file.BaseName

        $destFile = Join-Path `
            $outputRoot `
            "$safeName-crawler_url_image.txt"

        Copy-Item `
            $engineOutputFile `
            $destFile `
            -Force

        Write-Host "Saved:"
        Write-Host $destFile
    }
    else
    {
        Write-Warning "Output file not found!"
    }
}

Write-Host ""
Write-Host "ALL KEYWORDS FINISHED"