import numpy as np
import astropy
from astropy.time import Time
from astropy import constants as const
import astropy.units as u
from astropy.coordinates import Longitude, Latitude, Angle
from astropy import coordinates
import HW2.main as matrix

def sec2rad(deg=0.0, minute=0.0, sec=0.0):
    return np.pi * (deg * 3600 + minute * 60 + sec) / (180 * 3600)


def rotation_matrix(axis, theta):
    if axis == 'x': return np.array([[1, 0, 0],
                                     [0, np.cos(theta), np.sin(theta)],
                                     [0, -np.sin(theta), np.cos(theta)]
                                     ])

    if axis == 'y': return np.array([[np.cos(theta), 0, -np.sin(theta)],
                                     [0, 1, 0],
                                     [np.sin(theta), 0, np.cos(theta)]
                                     ])

    if axis == 'z': return np.array([[np.cos(theta), np.sin(theta), 0],
                                     [-np.sin(theta), np.cos(theta), 0],
                                     [0, 0, 1]
                                     ])

for i in range(3600):
    utc_time = '2022-03-16T11:06:00'
    time = Time(utc_time, scale='utc')
    # transform_matrix = matrix.CoordinatesTransformEquinoxBase(time, mid_matrix=True)
    # matrix_list = transform_matrix.transform()
    # earth_rotation_matrix = matrix_list[2][1]
    # earth_rotation_matrix = np.array([[0.94315765, -0.33234569, 0.0],
    #                                   [0.33234569, 0.94315765, 0.0],
    #                                   [0.0, 0.0, 1.0]])
    # polar_motion_matrix = np.array([[1.0, 3.17279566e-13, 1.63467841e-07],
    #                                 [0.0, 1.0, -1.94092957e-06],
    #                                 [-1.63467841e-07, 1.94092957e-06, 1.0]])

    site_coordinates = np.array([118.78, 32.07, 0])
    object_coordinates = np.array([-915731.0, 5958096.0, 3050295.0]) + i * np.array([6090.2, 2849.3, 3718.7])
    # print(earth_rotation_matrix @ object_coordinates)
    # print(polar_motion_matrix @ earth_rotation_matrix @ object_coordinates)
    h_0 = 0
    earth_radius = const.R_earth.value
    r = np.linalg.norm(object_coordinates)
    site_lon = 118.78 * u.deg
    site_lat = 32.07 * u.deg
    site_coord = coordinates.EarthLocation.from_geodetic(site_lon, site_lat, 0 * u.m).value
    x, y, z = site_coord
    linalg = np.array([x, y, z])
    R = np.linalg.norm(linalg)
    l_1 = (object_coordinates - linalg) @ rotation_matrix('z', x - 180) @ rotation_matrix('y', y - 90)
    x_1, y_1, z_1 = l_1
    tan_h = z_1 / np.sqrt(x_1**2 + y_1**2)
    h = np.arctan(tan_h) * 180 / np.pi
    if h>=0:
        print(i)
        break
    print(h)
