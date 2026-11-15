# Export every slide of pitch/VanaLens.pptx to pitch/preview/slide_NN.png via PowerPoint COM.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
New-Item -ItemType Directory -Force "$here\preview" | Out-Null
$pp = New-Object -ComObject PowerPoint.Application
$pres = $pp.Presentations.Open("$here\VanaLens.pptx", -1, 0, 0)
foreach ($s in $pres.Slides) { $n = "{0:D2}" -f $s.SlideIndex; $s.Export("$here\preview\slide_$n.png", "PNG", 1600, 900) }
$pres.Close(); $pp.Quit()
"exported $($pres.Slides.Count) slides"
