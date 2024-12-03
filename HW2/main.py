import numpy as np
from astropy.time import Time
from astropy.utils import iers


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


class CoordinatesTransformEquinoxBase:
    def __init__(self, utc_time: Time, mid_matrix: bool = False):
        # self.time = Time(time_str, scale=time_scale)
        self.time = utc_time
        self.delta_mu = 0.0
        self.mid_matrix = mid_matrix
        self.matrix_list = list()

    def precession_matrix(self):

        T = (self.time.tt.mjd - 51544.5) / 36525.0

        xi = 2306.2181 * T + 0.30188 * (T ** 2) + 0.017998 * (T ** 3)
        z = 2306.2181 * T + 1.09468 * (T ** 2) + 0.018203 * (T ** 3)
        theta = 2004.3109 * T - 0.42665 * (T ** 2) - 0.041833 * (T ** 3)

        xi = sec2rad(sec=xi)
        z = sec2rad(sec=z)
        theta = sec2rad(sec=theta)

        return rotation_matrix('z', -z) @ rotation_matrix('y', theta) @ rotation_matrix('z', -xi)

    def nutation_matrix(self):
        T = (self.time.tt.mjd - 51544.5) / 36525.0

        params = [[0, 0, 0, 0, 1, -171996, -174.2, 92025, 8.9],
                  [0, 0, 2, -2, 2, -13187, -1.6, 5736, -3.1],
                  [0, 0, 2, 0, 2, -2274, -0.2, 977, -0.5],
                  [0, 0, 0, 0, 2, 2062, 0.2, -895, 0.5],
                  [0, 1, 0, 0, 0, 1426, -3.4, 54, -0.1],
                  [1, 0, 0, 0, 0, 712, 0.1, -7, 0.0],
                  [0, 1, 2, -2, 2, -517, 1.2, 224, -0.6],
                  [0, 0, 2, 0, 1, -386, -0.4, 200, 0.0],
                  [1, 0, 2, 0, 2, -301, 0.0, 129, -0.1],
                  [0, -1, 2, -2, 2, 217, -0.5, -95, 0.3],
                  [1, 0, 0, -2, 0, -158, 0.0, -1, 0.0],
                  [0, 0, 2, -2, 1, 129, 0.1, -70, 0.0],
                  [-1, 0, 2, 0, 2, 123, 0.0, -53, 0.0],
                  [1, 0, 0, 0, 1, 63, 0.1, -33, 0.0],
                  [0, 0, 0, 2, 0, 63, 0.0, -2, 0.0],
                  [-1, 0, 2, 2, 2, -59, 0.0, 26, 0.0],
                  [-1, 0, 0, 0, 1, -58, -0.1, 32, 0.0],
                  [1, 0, 2, 0, 1, -51, 0.0, 27, 0.0],
                  [2, 0, 0, -2, 0, 48, 0.0, 1, 0.0],
                  [-2, 0, 2, 0, 1, 46, 0.0, -24, 0.0]]

        alpha_with_T = np.array([
            sec2rad(134, 57, 46.733) + (sec2rad(deg=1325 * 360) + sec2rad(198, 52, 2.633)) * T + sec2rad(sec=31.310) * (
                    T ** 2),
            sec2rad(357, 31, 39.804) + (sec2rad(deg=99 * 360) + sec2rad(359, 3, 1.224)) * T - sec2rad(sec=0.577) * (
                    T ** 2),
            sec2rad(93, 16, 18.977) + (sec2rad(deg=1342 * 360) + sec2rad(82, 2, 3.137)) * T - sec2rad(sec=13.257) * (
                    T ** 2),
            sec2rad(297, 51, 1.307) + (sec2rad(deg=1236 * 360) + sec2rad(307, 6, 41.328)) * T - sec2rad(sec=6.891) * (
                    T ** 2),
            sec2rad(125, 2, 40.280) - (sec2rad(deg=5 * 360) + sec2rad(134, 8, 10.539)) * T + sec2rad(sec=7.455) * (
                    T ** 2)
        ])

        delta_psi = 0.0
        delta_epsilon = 0.0
        for i in range(20):
            angle = 0
            for j in range(5):
                angle = angle + params[i][j] * alpha_with_T[j]
            delta_psi = delta_psi + (sec2rad(sec=params[i][5] * 1e-4) + sec2rad(sec=params[i][6] * 1e-4) * T) * np.sin(
                angle)
            delta_epsilon = delta_epsilon + (
                    sec2rad(sec=params[i][7] * 1e-4) + sec2rad(sec=params[i][8] * 1e-4) * T) * np.cos(angle)

        epsilon_A = sec2rad(deg=23, minute=26, sec=21.448) - sec2rad(sec=46.815) * T
        delta_mu = delta_psi * np.cos(epsilon_A)
        self.delta_mu = delta_mu
        delta_theta = delta_psi * np.sin(epsilon_A)

        return rotation_matrix('x', -delta_theta) @ rotation_matrix('y', delta_theta) @ rotation_matrix('z', -delta_mu)

    def earth_rotation_matrix(self):
        t = (self.time.ut1.mjd - 51544.5) / 36525.0
        S_G_hat = 18.6973746 + 879000.0513367 * t + (0.093104 * (t ** 2) - 6.2e-6 * (t ** 3) / 3600)
        S_G_hat_theta = S_G_hat * 2 * np.pi / 24.0
        return rotation_matrix('z', S_G_hat_theta + self.delta_mu)

    def polar_motion_matrix(self):
        iers_table = iers.IERS_Auto.open()
        x_p, y_p = iers_table.pm_xy(self.time)
        # x_p = iers_table.pm_x(utc_time)
        # y_p = iers_table.pm_y(utc_time)
        return rotation_matrix('y', -x_p) @ rotation_matrix('x', -y_p)

    def transform(self):
        final_matrix = self.polar_motion_matrix() @ self.earth_rotation_matrix() @ self.nutation_matrix() @ self.precession_matrix()
        if self.mid_matrix:
            self.matrix_list.append(('岁差矩阵PR', self.precession_matrix()))
            self.matrix_list.append(('章动矩阵NR', self.nutation_matrix()))
            self.matrix_list.append(('地球自转矩阵ER', self.earth_rotation_matrix()))
            self.matrix_list.append(('极移矩阵EP', self.polar_motion_matrix()))
            self.matrix_list.append(('最终矩阵', final_matrix))
        else:
            self.matrix_list.append(('最终矩阵', final_matrix))
        return self.matrix_list


class CoordinatesTransformCIOBase:
    def __init__(self, time_str: str, time_scale: str):
        self.time = Time(time_str, scale=time_scale)

    def matrix_cio(self):
        T = (self.time.tt.mjd - 51544.5) / 36525.0
        X = 9.7166e-3 * T - 3.32e-5 * np.sin(2.182 - 33.76 * T)
        Y = 4.46e-5 * np.cos(2.182 - 33.76 * T)
        a = 0.5 + (X ** 2 + Y ** 2) / 8
        M_sigma = np.array([[1 - a * X * X, -a * X * Y, -X],
                            [-a * X * Y, 1 - a * Y * Y, -Y],
                            [X, Y, 1 - a * (X * X + Y * Y)]])

        s = -0.5 * X * Y + sec2rad(sec=94e-6 + 3808.65e-6 * T)
        ERA = np.pi * 2 * (0.7790572732640 + 1.00273781191135448 * (self.time.ut1.mjd - 51544.5))

        return rotation_matrix('z', ERA) @ rotation_matrix('z', -s) @ M_sigma

    def transform(self):
        return self.matrix_cio()


utc = '2022-03-16T11:06:00'
time = Time(utc, scale='utc')

# 基于春分点的坐标转换
trans_equinox = CoordinatesTransformEquinoxBase(time, mid_matrix=True)

matrix_list = trans_equinox.transform()

# 输出结果矩阵和所有中间矩阵
for index in range(len(matrix_list)):
    name, matrix = matrix_list[index]
    print(name)
    print(matrix)

matrix_name, equ_final_matrix = matrix_list[-1]

# 基于CIO的坐标转换
trans_cio = CoordinatesTransformCIOBase(utc, 'utc')
print("基于CIO的坐标转换矩阵")
print(trans_cio.transform())

# 两种转换矩阵的差
cio_final_matrix = trans_cio.transform()
print("两种坐标转换矩阵的差")
print(equ_final_matrix - cio_final_matrix)
#
# print(trans_equinox.transform() - trans_cio.transform())
# print(polar_motion_matrix(time))
# # print(time)
# # print(nutation_matrix(time))
# T = (Time(time, scale='tt').mjd - 51544.5) / 36525.0
# # print(-8.34e-5 * np.sin(2.18 - 33.76 * T))
