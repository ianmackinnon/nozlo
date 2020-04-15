# Nozlo

G-code viewer for visualising feedrate

![Nozlo 3DBenchy screenshot](nozlo-3dbenchy.png)


## Installation

Nozlo requires Python 3.7 and an OpenGL-compatible display driver.

```
python3 -m pip install -e git+https://github.com/ianmackinnon/nozlo#egg=nozlo
```


## Usage

```
usage: nozlo [-h] [--verbose] [--quiet] [--version] [--layer LAYER] [--single]
             [--ortho] [--aim AIM AIM AIM] [--yaw YAW] [--pitch PITCH]
             [--dist DIST]
             GCODE

G-code viewer.

positional arguments:
  GCODE                 Path to G-code file.

optional arguments:
  -h, --help            show this help message and exit
  --verbose, -v         Print verbose information for debugging.
  --quiet, -q           Suppress warnings.
  --version, -V         show program's version number and exit
  --layer LAYER, -l LAYER
                        Layer number to display.
  --single, -s          Show only the single current layer.
  --ortho, -o           Use orthographic projection.
  --aim AIM AIM AIM, -a AIM AIM AIM
                        Aim coordinates for the camera.
  --yaw YAW, -Y YAW     Camera yaw in degrees, starting in positive X and
                        moving clockwise (looking down) around the aim point.
  --pitch PITCH, -P PITCH
                        Camera pitch in degrees, positive puts the camera
                        above the aim point.
  --dist DIST, -D DIST  Camera distance in mm for perspective view. Camera
                        frame in mm for orthographic view
```


## Controls

Mouse:

-   **Left drag**: Tumble camera
-   **Right drag**: Dolly camera
-   **Scroll**: Zoom camera

Keyboard

-   **Home**: toggle between first model layer and first movement layer
-   **End**: toggle between last model layer and first movement layer
-   **Up**: go up a layer
-   **Down**: go down a layer
-   **F**: frame model in camera
-   **A**: frame build surface in camera
-   **S**: toggle showing single layer
-   **O**: toggle orthographic/perspective projection

-   **Q**/**Escape**: Quit

