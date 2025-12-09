Geodesic contour extraction using fast marching method.
Computes distance maps from source points and extracts isolines.

Original project here: https://github.com/krummrey/contour-drawing/blob/main/contour-drawing.py
inspiration and thanks to https://www.reddit.com/user/docricky/ and https://www.reddit.com/user/krummrey/
for sharing Fast Marching Algorithm info

**input:**
image file (jpg, png etc)
  
**output:**
svg file with contour path segments

**example run commands:**

`input.jpg -o output.svg`

`input.jpg --source 100,200 --source 300,400 --num 50`

`input.jpg --source 100,200,300,400,500,600 --num 50`

`input.jpg --gamma 1.5 --blur 2.0 --thickness 2`

`input.jpg --dark-boost 2.0 --bright-cut 0.7 --num 120`

**this version contains fixes and updates:**

- addressed matplotlib QuadContourSet issue. Using cs.get_paths() instead of .collections
- added `smooth` parameter for smoothing output lines
- addressed matplotlib MOVETO - now split_path_at_moves() segments lines to prevent unwanted segments
- swapped in optional scikit-fmm (C) library to speed up Fast Marching Algorithm execution ~20-50x faster that plain python

**Input Image:**

<img width="600" height="1232" alt="dog" src="https://github.com/user-attachments/assets/2965395b-fb78-4371-b1b0-ae50a08d1675" />

**Output**:

command used:
`dog.png --source 100,200 --source 300,400 --num 200 --smooth 0 -o output_pass_1_new.svg`

<img width="600" height="1482" alt="contourTool_original_tweaked" src="https://github.com/user-attachments/assets/50d095c1-0a1c-4357-8902-3d99f40892eb" />
