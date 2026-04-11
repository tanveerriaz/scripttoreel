# Logo spec sheet crops

Source raster: `logo-spec-sheet.png` (copy of `Downloads/Logo Specs.png`).

Original dimensions for the file used to derive boxes: **2816×1536** px.

Crops are defined as proportional fractions of `(width, height)` so you can re-apply if the source image is re-exported at another resolution:

| Asset | left | top | right | bottom |
|-------|------|-----|-------|--------|
| `guidelines/primary_stacked.png` | 2% | 8% | 48% | 88% |
| `guidelines/icon_1024_panel.png` | 62% | 2% | 82% | 42% |
| `guidelines/legibility.png` | 82% | 2% | 99% | 38% |
| `guidelines/clear_space.png` | 62% | 38% | 99% | 68% |
| `guidelines/reverse_mono.png` | 62% | 68% | 99% | 98% |

Absolute boxes for **2816×1536**:

- primary_stacked: `(56, 122, 1351, 1351)`
- icon_1024_panel: `(1745, 30, 2309, 645)`
- legibility: `(2309, 30, 2787, 583)`
- clear_space: `(1745, 583, 2787, 1044)`
- reverse_mono: `(1745, 1044, 2787, 1505)`

Regenerate with:

```bash
python3 -c "
from PIL import Image
from pathlib import Path
ROOT = Path('assets/branding')
im = Image.open(ROOT / 'source/logo-spec-sheet.png')
w, h = im.size
# ... same fractions as table ...
"
```
