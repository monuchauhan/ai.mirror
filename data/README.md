# Sample data directory structure

Place your input files here:

## Input Videos
- `input/sample.mp4` - Sample video for frame extraction

## Garment Images
- `input/shirt_front.jpg` - Front view of garment
- `input/shirt_back.jpg` - Back view of garment

## Output Structure
After processing:
```
output/
├── frame_0001.png
├── frame_0002.png
├── ...
└── debug/ (if --debug enabled)
    └── frame_0001/
        ├── mask.png
        └── warped_garment.png
```

## Example Data
You can download sample fashion images from:
- [Deep Fashion Dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
- [VITON-HD Dataset](https://github.com/shadow2496/VITON-HD)
