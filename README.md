# SEM-Drawmaton

The SEM-Drawmaton is a completely mechanical drawing machine capable of drawing any continuous line drawing. The SEM-Drawmaton utilizes two rotors (cams) that displace a set of linkages to accurately trace out a desired image. The SEM-Drawmaton is based on Da Vinci's Drawmaton https://www.drawmaton.com/#/.

## Overview

This repository contains code to compute rotor profiles necessary to draw a given image or line drawing. The application supports various input types:
- Image files (.jpg, .png, etc)
- SVG files
- Parametric equations
- Ordered X,Y coordinates

The process involves:
1. Converting input to X,Y coordinates
2. Calculating Theta values (linkage angles) using SciPy.optimize's fsolve
3. Computing rotor radii
4. Storing simulation data (dimensions, coordinates, theta values, radii)

## Command Line Interface

The application provides a command-line interface (CLI) for all operations:

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```bash
# Create simulation from image (non-interactive)
python -m cli create image -i input.jpg -o simulation.txt

# Create simulation with interactive contour selection
python -m cli create image -i input.jpg -o simulation.txt --interactive

# Create simulation from SVG
python -m cli create svg -i input.svg -o simulation.txt

# Show animation
python -m cli animate -i simulation.txt

# Export animation to GIF
python -m cli export-animation -i simulation.txt -o animation.gif

# Export rotor profiles
python -m cli export-rotors -i simulation.txt -b bottom.svg -t top.svg

# Show rotor gaps
python -m cli show-gaps -i simulation.txt
```

### Common Options
All dimension parameters can be customized:
```bash
python -m cli create image -i input.jpg -o sim.txt \
    --l1 5.8 --l2 16.6 --l3 24.3 --gx -4.5 --gy 16.0
```

### Image Processing Options
```bash
python -m cli create image -i input.jpg -o sim.txt \
    --target-x 13.4 --target-y 27.1 \
    --target-w 16.3 --target-h 16.3
```

## Implementation Details

The application consists of several key components:
- `main.py`: Entry point for script-based usage
- `cli.py`: Command-line interface implementation
- `SimulateDrawmaton.py`: Core simulation and animation logic
- `ImageToXY.py`, `SVGToXY.py`: Input processing
- `utilities.py`: Analysis and export functions

This computational model was developed in close conjunction with a Fusion 360 CAD model of the SEM-Drawmaton.
