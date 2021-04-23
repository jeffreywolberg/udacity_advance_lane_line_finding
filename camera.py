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

    def find_camera_calib(self):

        objpoints = []
        imgpoints = []

        nx, ny = 9, 6
        checkerboard_points = np.zeros(shape=(nx*ny, 3), dtype=np.float32)
        checkerboard_points[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        for image_path in tqdm.tqdm(glob.glob(join(self.images_folder, 'calibration*.jpg'))):
            # print(image_path)
            image = cv2.imread(image_path)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, img_corners = cv2.findChessboardCorners(gray, (nx,ny), None)
            if ret == True:
                objpoints.append(checkerboard_points)
                imgpoints.append(img_corners)
                img = cv2.drawChessboardCorners(image, (nx, ny), img_corners, ret)
                img_corners = img_corners.reshape(-1, 2).astype(np.float32)

                # print(img_corners.shape, checkerboard_points.shape)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (nx, ny), img_corners, ret)
            # print(test_image.shape[1::-1])

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, test_image.shape[1::-1], None, None)
        return ret, mtx, dist, rvecs, tvecs

    def show_undistorted_image(self, image, mtx, dist):
        dst = cv2.undistort(image, mtx, dist, None, mtx)
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(test_image)
        axarr[1].imshow(dst)
        plt.savefig(join('.', 'output_images', 'undistorted_img_comparison.jpg'))
        plt.show()

if __name__ == '__main__':
    images_folder = r'.\camera_cal'
    test_image = cv2.imread(join(images_folder, 'calibration1.jpg'))
    camera = Camera(images_folder)
    ret, mtx, dist, rvecs, tvecs = camera.find_camera_calib()

    # images = glob.glob(join(image_folder, 'calibration*.jpg'))
    # for test_image_path in images:
    #     test_image = cv2.imread(test_image_path)
    camera.show_undistorted_image(test_image, mtx, dist)


