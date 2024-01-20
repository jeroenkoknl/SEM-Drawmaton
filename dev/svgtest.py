import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib.path import Path
import numpy as np
from xml.dom import minidom
from svg.path import parse_path

def ParseSVG(filepath):
    doc = minidom.parse(filepath)
    
    svg_element = doc.getElementsByTagName('svg')[0]
    view_box = svg_element.getAttribute('viewBox')
    view_box_values = [float(value) for value in view_box.split()]
    width, height = view_box_values[2], view_box_values[3]
    print(f"Width: {width}, Height: {height}")

    path = doc.getElementsByTagName('path')[0]
    d = path.getAttribute('d')
    parsed = parse_path(d)
    # print(f'Objects ({len(parsed)}):\n', parsed, '\n' + '-' * 20)
    # for obj in parsed:
    #     if(type(obj).__name__ == 'CubicBezier'):
    #         print(type(obj).__name__, ', start: ', (round(obj.start.real, 3), round(obj.start.imag, 3)), ' , control1: ', (round(obj.control1.real, 3), round(obj.control1.imag, 3)), ' , control2: ', (round(obj.control2.real, 3), round(obj.control2.imag, 3)), ' , end:', (round(obj.end.real, 3), round(obj.end.imag, 3)))
    #     else:
    #         print(type(obj).__name__, ', start/end coords:', ((round(obj.start.real, 3), round(obj.start.imag, 3)), (round(obj.end.real, 3), round(obj.end.imag, 3))))
    # print('-' * 20)
    doc.unlink()
    
    return width, height, parsed

def SVGtoCoords(width, height, parsed, pointspercurve):
    numcurves = len(parsed) - 2
    n = numcurves*pointspercurve - numcurves + 1
    print(f'numcurves = {numcurves}, n = {n}')
    coords = np.zeros((2, int(n)))
    
    startind = 0
    for obj in parsed:
        if(type(obj).__name__ == 'CubicBezier'):
            coords[:, startind:startind+pointspercurve] = interpolate_cubic_bezier([obj.start.real, obj.start.imag], [obj.control1.real, obj.control1.imag], [obj.control2.real, obj.control2.imag], [obj.end.real, obj.end.imag], pointspercurve)
            print(type(obj).__name__, ', start: ', (round(obj.start.real, 3), round(obj.start.imag, 3)), ' , control1: ', (round(obj.control1.real, 3), round(obj.control1.imag, 3)), ' , control2: ', (round(obj.control2.real, 3), round(obj.control2.imag, 3)), ' , end:', (round(obj.end.real, 3), round(obj.end.imag, 3)))
            startind += pointspercurve - 1
        if(type(obj).__name__ == 'Line'):
            coords[:, startind:startind+pointspercurve] = interpolate_line([obj.start.real, obj.start.imag], [obj.end.real, obj.end.imag], pointspercurve)
            print(type(obj).__name__, ', start: ', (round(obj.start.real, 3), round(obj.start.imag, 3)), ' , end:', (round(obj.end.real, 3), round(obj.end.imag, 3)))
            startind += pointspercurve - 1
        else:
            print(type(obj).__name__, ', start/end coords:', ((round(obj.start.real, 3), round(obj.start.imag, 3)), (round(obj.end.real, 3), round(obj.end.imag, 3))))
    print(coords)
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

def PositionImage(xcoords, ycoords, brectx, brecty, brectw, brecth, targetx, targety, targetw, targeth):
    wscale = targetw / brectw
    hscale = targeth / brecth
    if (wscale <= hscale):
        # print('hi') 
        xcoords,ycoords = scale(wscale, xcoords, ycoords, brectw, brecth)
    else:
        xcoords,ycoords = scale(hscale, xcoords, ycoords, brectw, brecth)
        
    xcoords += targetx - np.min(xcoords)
    ycoords += targety - np.max(ycoords)
    
    return xcoords, ycoords
    
def scale(factor, xcoords, ycoords, currw, currh):
    neww = currw*factor
    newh = currh*factor
    for i in range(len(xcoords)):
        xcoords[i] *= factor
        ycoords[i] *= factor
    return [xcoords, ycoords]    
   

def PixcoordsToXY(L1, L2, L3, Gx, Gy, targetx, targety, targetw, targeth, pixcoords, width, height):
    
    pixcoords[1,:] = height - pixcoords[1,:]
    brectx, brecty, brectw, brecth = BoundingRectangle(pixcoords)
    
    fig, ax = plt.subplots()
    # ax.plot(xcoords, ycoords)
    # print(brectx, brecty, brectw, brecth)

    adjxcoords, adjycoords = PositionImage(pixcoords[0,:], pixcoords[1,:], brectx, brecty, brectw, brecth, targetx, targety, targetw, targeth)
    
    ax.plot(adjxcoords, adjycoords)
    ax.scatter([0,Gx], [0, Gy])
    refcirc = plt.Circle((0, 0), radius=L2 + L3, fill=False)
    brect = Rectangle((brectx,brecty), brectw, -brecth, color="orange", fill=False)
    adjrect = Rectangle((targetx, targety), targetw, -targeth, color="green", fill=False)
    ax.add_patch(brect)
    ax.add_patch(adjrect)
    ax.set_xlim(-20, 50)
    ax.set_ylim(-20, 50)
    ax.set_aspect('equal')
    plt.gca().add_artist(refcirc)
    plt.grid()
    plt.show()
    # print(adjxcoords, adjycoords)
    return adjxcoords, adjycoords



filepath = './InputImages/catAI.svg'
width, height, parsed = ParseSVG(filepath)

pointspercurve = 5
# start_point = [100.83, 275.66]
# control_point1 = [103.65, 271.52]
# control_point2 = [120.18, 255.79]
# end_point = [143.94, 253.19]
# print(interpolate_cubic_bezier(start_point, control_point1, control_point2, end_point, pointspercurve), '\n')
# print(interpolate_cubic_bezier([143.94, 253.19], [163.34, 251.07], [186.46, 257.82], [187.05, 265.57], pointspercurve))
print('-'*20)
pixcoords = SVGtoCoords(width, height, parsed, pointspercurve)

L1 = 4.1
L2 = 16.2
L3 = 24.0
Gx = -4.0
Gy = 15.0
targetx = 13
targety = 26
targetw = 16
targeth = 16 
# PixcoordsToXY(L1, L2, L3, Gx, Gy, targetx, targety, targetw, targeth, pixcoords, width, height)
fig, ax = plt.subplots()
ax.plot(pixcoords[0,:], height - pixcoords[1,:], ls='-', marker='.')
ax.set_aspect('equal')
plt.show()
