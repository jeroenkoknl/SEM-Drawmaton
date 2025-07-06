import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib.path import Path
import numpy as np
from xml.dom import minidom
from svg.path import parse_path
import ImageToXY as imgToXY

def ParseSVG(filepath):
    doc = minidom.parse(filepath)
    
    svg_element = doc.getElementsByTagName('svg')[0]
    view_box = svg_element.getAttribute('viewBox')
    view_box_values = [float(value) for value in view_box.split()]
    width, height = view_box_values[2], view_box_values[3]
    # print(f"Width: {width}, Height: {height}")

    path = doc.getElementsByTagName('path')[0]
    d = path.getAttribute('d')
    parsed = parse_path(d)
    doc.unlink()
    
    return width, height, parsed

def SVGtoPixelCoords(width, height, parsed, pointspercurve):
    numcurves = len(parsed) - 2
    # n = numcurves*pointspercurve - numcurves + 1
    # print(f'numcurves = {numcurves}, n = {n}')
    startind = 0
    for obj in parsed:
        if (type(obj).__name__ == 'CubicBezier' or type(obj).__name__ == 'Line'):
            startind += pointspercurve - 1
    # print(startind)
    n = startind + 1
    
    coords = np.zeros((2, n))
        
    startind = 0
    for obj in parsed:
        if(type(obj).__name__ == 'CubicBezier'):
            coords[:, startind:startind+pointspercurve] = interpolate_cubic_bezier([obj.start.real, obj.start.imag], [obj.control1.real, obj.control1.imag], [obj.control2.real, obj.control2.imag], [obj.end.real, obj.end.imag], pointspercurve)
            # print(type(obj).__name__, ', start: ', (round(obj.start.real, 3), round(obj.start.imag, 3)), ' , control1: ', (round(obj.control1.real, 3), round(obj.control1.imag, 3)), ' , control2: ', (round(obj.control2.real, 3), round(obj.control2.imag, 3)), ' , end:', (round(obj.end.real, 3), round(obj.end.imag, 3)))
            startind += pointspercurve - 1
        elif(type(obj).__name__ == 'Line'):
            coords[:, startind:startind+pointspercurve] = interpolate_line([obj.start.real, obj.start.imag], [obj.end.real, obj.end.imag], pointspercurve)
            # print(type(obj).__name__, ', start: ', (round(obj.start.real, 3), round(obj.start.imag, 3)), ' , end:', (round(obj.end.real, 3), round(obj.end.imag, 3)))
            startind += pointspercurve - 1
        # else:
            # print('hi')
            # print(type(obj).__name__, ', start/end coords:', ((round(obj.start.real, 3), round(obj.start.imag, 3)), (round(obj.end.real, 3), round(obj.end.imag, 3))))
    # print(coords)
    return coords
    

def interpolate_cubic_bezier(start, control1, control2, end, n):
    t = np.linspace(0, 1, n)
    B = np.zeros((2,n))
    B[0] = ((1-t)**3) * start[0] + (3*t*(1-t)**2)*control1[0] + (3*(1-t)*t**2)*control2[0] + (t**3)*end[0] 
    B[1] = ((1-t)**3) * start[1] + (3*t*(1-t)**2)*control1[1] + (3*(1-t)*t**2)*control2[1] + (t**3)*end[1]
    return B 

def interpolate_line(start, end, n):
    t = np.linspace(0, 1, n)
    B = np.zeros((2,n))
    B[0] = start[0] + t*(end[0] - start[0])
    B[1] = start[1] + t*(end[1] - start[1])
    return B

def BoundingRectangle(coords):
    minx = np.min(coords[0,:])
    maxx = np.max(coords[0,:])
    miny = np.min(coords[1,:])
    maxy = np.max(coords[1,:])
    w = maxx - minx
    h = maxy - miny
    return minx, miny, w, h
   

def SVGtoXY(L1, L2, L3, Gx, Gy, filepath, targetx, targety, targetw, targeth, pointspercurve=10, preview_mode='none'):
    width, height, parsedSVG = ParseSVG(filepath)
    pixcoords = SVGtoPixelCoords(width, height, parsedSVG, pointspercurve)
    pixcoords[1,:] = height - pixcoords[1,:]
    brectx, brecty, brectw, brecth = BoundingRectangle(pixcoords)
    
    adjxcoords, adjycoords = imgToXY.PositionImage(pixcoords[0,:], pixcoords[1,:], brectx, brecty, brectw, brecth, targetx, targety, targetw, targeth)
    
    util.show_preview(
        coords=(adjxcoords, adjycoords),
        dims=(L1, L2, L3, Gx, Gy),
        title="SVG Preview",
        preview_mode=preview_mode,
        close_delay=3
    )
    
    return adjxcoords, adjycoords
