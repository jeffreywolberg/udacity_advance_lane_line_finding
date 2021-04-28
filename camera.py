import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import tqdm

join = os.path.join

class Camera(object):
    def __init__(self, images_folder):
        self.images_folder = images_folder
        self.mtx = None
        self.dist = None

        self.mtx = np.array([[1.15777930e+03, 0.00000000e+00, 6.67111054e+02],
 [0.00000000e+00, 1.15282291e+03, 3.86128938e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],])
        self.dist = np.array([[-0.24688775, -0.02373133, -0.00109842,  0.00035108, -0.00258571]])

    def get_mtx_and_dist(self):
        if self.mtx is None or self.dist is None:
            ret, self.mtx, self.dist, rvecs, tvecs = self.find_camera_calib()
        # print("Matrix:", self.mtx)
        # print("Dist:", self.dist)
        return self.mtx, self.dist

    def find_camera_calib(self):
        if self.mtx is not None and self.dist is not None:
            return 0, self.mtx, self.dist, 0, 0

        objpoints = []
        imgpoints = []

        nx, ny = 9, 6
        checkerboard_points = np.zeros(shape=(nx*ny, 3), dtype=np.float32)
        checkerboard_points[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        image_shape = None
        print("Calibrating Camera ...")
        for image_path in tqdm.tqdm(glob.glob(join(self.images_folder, 'calibration*.jpg'))):
            # print(image_path)
            image = cv2.imread(image_path)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, img_corners = cv2.findChessboardCorners(gray, (nx,ny), None)
            if ret == True:
                if image_shape is None:
                    image_shape = gray.shape[1::-1]
                objpoints.append(checkerboard_points)
                imgpoints.append(img_corners)
                img = cv2.drawChessboardCorners(image, (nx, ny), img_corners, ret)
                img_corners = img_corners.reshape(-1, 2).astype(np.float32)

                # print(img_corners.shape, checkerboard_points.shape)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (nx, ny), img_corners, ret)
            # print(test_image.shape[1::-1])

        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)
        return ret, self.mtx, self.dist, rvecs, tvecs

    def undistort_image(self, image, mtx, dist):
        dst = cv2.undistort(image, mtx, dist, None, mtx)
        return dst
        # f, axarr = plt.subplots(1,2)
        # axarr[0].imshow(test_image)
        # axarr[1].imshow(dst)
        # plt.savefig(join('.', 'output_images', 'undistorted_img_comparison.jpg'))
        # plt.show()

if __name__ == '__main__':
    images_folder = r'.\camera_cal'
    test_image = cv2.imread(join(images_folder, 'calibration1.jpg'))
    camera = Camera(images_folder)
    ret, mtx, dist, rvecs, tvecs = camera.find_camera_calib()

    # images = glob.glob(join(image_folder, 'calibration*.jpg'))
    # for test_image_path in images:
    #     test_image = cv2.imread(test_image_path)
    camera.undistort_image(test_image, mtx, dist)


