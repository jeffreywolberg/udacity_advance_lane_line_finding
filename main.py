import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

join = os.path.join

from camera import Camera

class LaneFinder(object):
    def __init__(self, camera_obj, image=None, video=None, debug=False):
        self.camera_obj = camera_obj
        self.mtx, self.dist = camera.get_mtx_and_dist()
        self.image = image
        self.video = video
        self.polynomial_coeffs = []
        self.debug = debug
        self.frame_num = 0
        self.show_every_nth_frame = 10

        # set in case you'd like sliding boxes to be tried after frames_since_reset conseuctive frames of using poly to
        # refit polynomial coefficients
        self.frames_since_reset = 1
        self.frames_to_reset = 100000

        self.lanes_found_w_boxes = 0
        self.lanes_found_w_poly_refit = 0
        self.conditions = np.zeros(shape=(3,))
        self.conditions_counter = np.zeros(shape=(3,))
        # cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)

#===================== IMAGE PROCESSING/DRAWING =================================
    def transform_roi(self, image):
        y_top = image.shape[0] - 240
        lx_small, lx_large, rx_small, rx_large = 225, image.shape[1] // 2 - 125, image.shape[1] // 2 + 160, image.shape[1] - 70
        top_left = [lx_large, y_top]
        top_right = [rx_small, y_top]
        bottom_left = [lx_small, image.shape[0]]
        bottom_right = [rx_large, image.shape[0]]
        vertices = np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32)

        tl, tr, bl, br = [0, 0], [image.shape[1], 0], [0, image.shape[0]], [image.shape[1], image.shape[0]]
        # tl, tr, bl, br = [(image.shape[0] / 4), 0], [(image.shape[0] * 3 / 4), 0], [(image.shape[0] / 4), image.shape[0]], [(image.shape[0] * 3 / 4), image.shape[0]]
        dst_points = np.array([tl, tr, bl, br], dtype=np.float32)

        if self.debug and self.frame_num % self.show_every_nth_frame == 0:
            image_drawn = self.draw_manual_lines(vertices, image.copy())
            cv2.imshow('ROI', image_drawn)
            cv2.waitKey(0)

        image_warped, M = self.warp_image(vertices, dst_points, image)

        return vertices, dst_points, image_warped

    def draw_manual_lines(self, vertices, image):
        cv2.line(image, (int(vertices[2, 0]), int(vertices[2, 1])), (int(vertices[0, 0]), int(vertices[0, 1])),
                 color=(255, 0, 0), thickness=3)
        cv2.line(image, (int(vertices[3, 0]), int(vertices[3, 1])), (int(vertices[1, 0]), int(vertices[1, 1])),
                 color=(255, 0, 0), thickness=3)
        for dot in vertices:
            cv2.circle(image, (int(dot[0]), int(dot[1])), 3, (0, 0, 255), 2)
        return image

    def get_binary_image(self, img, s_thresh=(170, 255), sx_thresh=(30, 100)):
        img = np.copy(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        h_channel = hls[:, :, 0]
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        # Sobel x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        # print(np.max(abs_sobelx), np.median(abs_sobelx))

        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # Stack each channel
        # color_binary = np.dstack((sxbinary, np.zeros_like(sxbinary), s_binary)) * 255

        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        if self.debug and self.frame_num % (self.show_every_nth_frame) == 0:
            cv2.imshow("Combined Binary", combined_binary.astype(np.float64))
            cv2.waitKey(0)
            # print(s_channel[700, 220:250])
            self.display_two_ims(sxbinary, s_binary, 'Sobel_x', 'S_channel')

        # cv2.imshow("Temp", combined_binary.astype(np.float64))
        # cv2.waitKey(0)
        return combined_binary

    def warp_image(self, vertices_src, vertices_dst, image):
        M = cv2.getPerspectiveTransform(vertices_src,
                                        vertices_dst)
        image_warped = cv2.warpPerspective(image, M, dsize=image.shape[1::-1])
        return image_warped, M

    def display_two_ims(self, im, im2, title, title2):
        # Plot the result
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()

        ax1.imshow(im)
        ax1.set_title(title, fontsize=30)

        ax2.imshow(im2)
        ax2.set_title(title2, fontsize=30)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
        # f.savefig(r'./output_images/warped_straight_lines.jpg')

    def image_col_peaks(self, binary_image_warped):

        midpoint_x = binary_image_warped.shape[1] // 2
        y_desired_height = 0

        pixel_sums = np.sum(binary_image_warped[y_desired_height:, :], axis=0)

        self.left_peak_x, val_l = np.argmax(pixel_sums[:midpoint_x]), np.max(pixel_sums[:midpoint_x])
        self.right_peak_x, val_r = np.argmax(pixel_sums[midpoint_x:]) + midpoint_x, np.max(pixel_sums[midpoint_x:])

        if self.debug and self.frame_num % (self.show_every_nth_frame) == 0:
            plt.plot(np.arange(binary_image_warped.shape[1]), pixel_sums)
            plt.plot([self.left_peak_x, self.left_peak_x], [0, val_l])
            plt.plot([self.right_peak_x, self.right_peak_x], [0, val_r])
            plt.show()
        # plt.savefig(r'./output_images/pixel_col_sum_graph.png')
        return self.left_peak_x, self.right_peak_x

    def find_lane_pixels(self, binary_warped, leftx_base, rightx_base, left_coeffs=None, right_coeffs=None, colored_image=None):
        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 80
        # Set minimum number of pixels found to recenter window
        minpix = 10
        # if sliding box moves this amount horizontally, don't collect any points from that box
        max_box_movement = 80

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        if left_coeffs is None or right_coeffs is None:
            did_use_boxes = True
            # t = time.time()
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window + 1) * window_height
                win_y_high = binary_warped.shape[0] - window * window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin

                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                  (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]

                good_right_inds = \
                ((nonzerox >= win_xright_low) & (nonzerox < win_xright_high) & (nonzeroy >= win_y_low) & (
                        nonzeroy < win_y_high)).nonzero()[0]

                show_l_rect = False
                if len(good_left_inds) > minpix:
                    l_center_box_diff = np.int(np.mean(nonzerox[good_left_inds])) - leftx_current
                    # if self.debug and self.frame_num % self.show_every_nth_frame == 0:
                        # print("L", window, l_center_box_diff, np.int(np.mean(nonzerox[good_left_inds])), leftx_current, len(good_left_inds))
                    if np.abs(l_center_box_diff) < max_box_movement:
                        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                        left_lane_inds.append(good_left_inds)
                        show_l_rect = True


                show_r_rect = False
                if len(good_right_inds) > minpix:
                    r_center_box_diff = np.int(np.mean(nonzerox[good_right_inds])) - rightx_current
                    # if self.debug and self.frame_num % self.show_every_nth_frame == 0:
                        # print("R", window, r_center_box_diff, np.int(np.mean(nonzerox[good_right_inds])), rightx_current, len(good_right_inds))
                    if np.abs(r_center_box_diff) < max_box_movement:
                        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                        right_lane_inds.append(good_right_inds)
                        show_r_rect = True




                # Draw the windows on the visualization image
                if self.debug and self.frame_num % (self.show_every_nth_frame) == 0:
                    if colored_image is not None:
                        if show_l_rect:
                            cv2.rectangle(colored_image, (win_xleft_low, win_y_low),
                                          (win_xleft_high, win_y_high), (255,0, 0), 2)
                        if show_r_rect:
                            cv2.rectangle(colored_image, (win_xright_low, win_y_low),
                                          (win_xright_high, win_y_high), (255,0, 0), 2)

            if self.debug and self.frame_num % (self.show_every_nth_frame) == 0:
                cv2.imshow("Boxes", colored_image)
                cv2.waitKey(0)

        else:
            did_use_boxes = False
            left_lane_inds = ((nonzerox > (left_coeffs[0] * (nonzeroy ** 2) + left_coeffs[1] * nonzeroy +
                                           left_coeffs[2] - margin)) & (nonzerox < (left_coeffs[0] * (nonzeroy ** 2) +
                                                                                    left_coeffs[1] * nonzeroy +
                                                                                    left_coeffs[
                                                                                        2] + margin)))
            right_lane_inds = ((nonzerox > (right_coeffs[0] * (nonzeroy ** 2) + right_coeffs[1] * nonzeroy +
                                            right_coeffs[2] - margin)) & (
                                           nonzerox < (right_coeffs[0] * (nonzeroy ** 2) +
                                                       right_coeffs[1] * nonzeroy + right_coeffs[
                                                           2] + margin)))

            left_curve_values, right_curve_values, ploty = self.get_poly_curve(binary_warped.shape, left_coeffs, right_coeffs)

            ## Visualization ##
            # Create an image to draw on and an image to show the selection window
            if self.debug and self.frame_num % (self.show_every_nth_frame) == 0:
                out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
                window_img = np.zeros_like(out_img)
                # Color in left and right line pixels
                out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
                out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
                # Generate a polygon to illustrate the search window area
                # And recast the x and y points into usable format for cv2.fillPoly()
                left_line_window1 = np.array([np.transpose(np.vstack([left_curve_values - margin, ploty]))])
                left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_curve_values + margin,
                                                                                ploty])))])
                left_line_pts = np.hstack((left_line_window1, left_line_window2))
                right_line_window1 = np.array([np.transpose(np.vstack([right_curve_values - margin, ploty]))])
                right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_curve_values + margin,
                                                                                 ploty])))])
                right_line_pts = np.hstack((right_line_window1, right_line_window2))
                print(left_line_pts.shape)
                # Draw the lane onto the warped blank image
                cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
                cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
                result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

                cv2.imshow("Poly adjust", result)
                # print(right_coeffs)
                cv2.waitKey(0)

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            if did_use_boxes:
                left_lane_inds = np.concatenate(left_lane_inds)
                right_lane_inds = np.concatenate(right_lane_inds)
            # print("NORMAL", left_lane_inds.shape, right_lane_inds.shape)
        except ValueError:
            print("Lane inds value error, l shape: {}, r_shape {}".format(len(left_lane_inds), len(right_lane_inds)))
            # print(len(left_lane_inds), len(right_lane_inds))
            # print("VALUE ERROR", left_lane_inds.shape==right_lane_inds.shape)
            return np.array([]), np.array([]), np.array([]), np.array([])
            # Avoids an error if the above is not implemented fully

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    def find_offset_from_center(self, leftx_bottom, rightx_bottom, image_shape, mx=1.0):
        position = (leftx_bottom + rightx_bottom) / 2
        pixel_offset = position - (image_shape[1] // 2)
        return pixel_offset, pixel_offset*mx

# ===========================POLYNOMIAL FITTING=========================================

    def get_poly_coeffs(self, leftx, lefty, rightx, righty, ym=1.0, xm=1.0):
        try:
            left_coeffs = np.polyfit(lefty*ym, leftx*xm, 2)
        except(TypeError):
            print('Type error no lefty or leftx')
            left_coeffs =  self.polynomial_coeffs[-1][0]
        try:
            right_coeffs = np.polyfit(righty*ym, rightx*xm, 2)
        except(TypeError):
            print('Type error no righty or rightx')
            # print(lefty.shape, leftx.shape, rightx.shape, righty.shape, rightx, righty)
            right_coeffs = self.polynomial_coeffs[-1][1]

        # rolling mean
        if len(self.polynomial_coeffs)>=10:
            left_coeffs = np.mean(np.array(self.polynomial_coeffs)[:10, 0], axis=0) * .5 + (.5 * np.array(left_coeffs))
            right_coeffs = np.mean(np.array(self.polynomial_coeffs)[:10, 1], axis=0) * .5 + (.5 * np.array(right_coeffs))

        return left_coeffs, right_coeffs

    def get_poly_curve(self, img_shape, left_coeffs, right_coeffs):
        '''
        Given an image shape, generate the curve based off of poly_coeffs, left_coeffs and right_coeffs
        '''
        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
        left_x = (left_coeffs[0] * (ploty ** 2)) + (left_coeffs[1] * ploty) + left_coeffs[2]
        right_x = (right_coeffs[0] * (ploty ** 2)) + (right_coeffs[1] * ploty) + right_coeffs[2]
        # x = mx / (my ** 2) * a * (y ** 2) + (mx / my) * b * y + c

        return left_x, right_x, ploty

    def measure_curvature(self, left_coeffs, right_coeffs, ploty, ym_per_pix=1.0):
        '''
        Calculates the curvature of polynomial functions in pixels. The poly coefficients must be adjusted
        to reflect meters instead of pixels
        '''
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        # Calculation of R_curve (radius of curvature)
        left_curverad = ((1 + (2 * left_coeffs[0] * ym_per_pix * y_eval + left_coeffs[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_coeffs[0])
        right_curverad = ((1 + (
                    2 * right_coeffs[0] * ym_per_pix * y_eval + right_coeffs[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_coeffs[0])

        return left_curverad, right_curverad

    def draw_lanes_and_shaded_green_area(self, l_line_points, r_line_points, Minv, line_color, image):
        blank = np.zeros_like(image)
        product = np.matmul(Minv, l_line_points.T).T
        rescale_num = product[:, 2, np.newaxis]
        rescale_num = np.repeat(rescale_num, 3, axis=1)
        l_line_points_unwarped = np.int32(np.divide(product, rescale_num)[:, :2])
        cv2.polylines(blank, np.int32([l_line_points_unwarped]), False, line_color, thickness=10)

        product = np.matmul(Minv, r_line_points.T).T
        rescale_num = product[:, 2, np.newaxis]
        rescale_num = np.repeat(rescale_num, 3, axis=1)
        r_line_points_unwarped = np.int32(np.divide(product, rescale_num)[:, :2])
        cv2.polylines(blank, np.int32([r_line_points_unwarped]), False, line_color, thickness=10)

        lines_points = np.hstack((l_line_points_unwarped, r_line_points_unwarped))
        lines_points = lines_points.reshape((-1, 2)) # makes even indices l point, odd indices r point

        cv2.fillPoly(blank, np.int32([lines_points]), (150, 255, 150))

        return blank

#===============================================PIPELINE====================================================

    def find_lane_lines(self, image):
        # with manually selected vertices of lane, take roi of car_image and
        # transform it into a bird's eye view of lane markers
        source_pts, dst_points, image_warped = self.transform_roi(image)

        # obtain binary image
        binary_image_warped = self.get_binary_image(image_warped)

        if self.debug and self.frame_num % self.show_every_nth_frame == 0:
            cv2.imshow("Warped Binary Im", binary_image_warped.astype(np.float64))
            cv2.waitKey(0)
            # cv2.imwrite(r'./output_images/warped_binary_image.jpg', binary_image_warped*255)
        # from binary image, find pixel peaks within columns
        self.left_peak_x, self.right_peak_x = self.image_col_peaks(binary_image_warped)
        # given the peaks, find the lane pixels within the binary warped image
        n_frames_compare_coeffs = 5
        # if there are more than n_frames_to_compare, and 'a' of polynomial coeff is not 10x greater/smaller than
        # 'a' coeff from n_frames_to_compare frames ago, use previous coeffs to refit curve
        # , else recompute line/coeffs from scratch
        try:
            self.conditions = np.array([(self.frames_since_reset < self.frames_to_reset), len(self.polynomial_coeffs) > n_frames_compare_coeffs, 10 > np.abs(self.polynomial_coeffs[-n_frames_compare_coeffs][0][0] / self.polynomial_coeffs[-1][0][0])  > .1])
            self.conditions_counter += self.conditions
            self.poly_refit = all(self.conditions)
        except(IndexError, TypeError):
            self.poly_refit = False
        if self.poly_refit:
            leftx, lefty, rightx, righty = self.find_lane_pixels(binary_image_warped, self.left_peak_x,
                                                                            self.right_peak_x,
                                                                            left_coeffs=self.polynomial_coeffs[-1][0],
                                                                            right_coeffs=self.polynomial_coeffs[-1][
                                                                                1],
                                                                 colored_image=image_warped.copy())
            self.lanes_found_w_poly_refit += 1
            self.frames_since_reset += 1
        else:
            leftx, lefty, rightx, righty = self.find_lane_pixels(binary_image_warped, self.left_peak_x,
                                                                                self.right_peak_x,
                                                                 colored_image=image_warped.copy())
            self.lanes_found_w_boxes += 1
            self.frames_since_reset = 1
            # print("RECOMPUTING FROM SCRATCH")

        # line_color = (255, 0, 0) if poly_refit else (0, 0, 255)
        line_color = (255, 0, 0)

        # after finding lane pixels, find a 2nd deg polynomial to fit the pixels for each r and l lines
        left_coeffs, right_coeffs = self.get_poly_coeffs(leftx, lefty, rightx, righty)
        self.polynomial_coeffs.append([left_coeffs, right_coeffs])
        # generate the x,y data for the curve
        left_points, right_points, ploty = self.get_poly_curve(image_warped.shape, left_coeffs, right_coeffs)
        l_line_points = (np.array([left_points, ploty, np.ones(left_points.shape[0])]).T).astype(np.int32)
        r_line_points = (np.array([right_points, ploty, np.ones(left_points.shape[0])]).T).astype(np.int32)
        # print(l_line_points.shape)

        Minv = cv2.getPerspectiveTransform(dst_points, source_pts)
        img_lines = self.draw_lanes_and_shaded_green_area(l_line_points, r_line_points, Minv, line_color, image)

        # after getting curve coordinates, draw line
        # blank_left, blank_right = np.zeros_like(image), np.zeros_like(image)
        # cv2.polylines(blank_left, [l_line_points[:, :2]], False, line_color, thickness=12)  # , color=(0,0,255))
        # cv2.polylines(blank_right, [r_line_points[:, :2]], False, line_color, thickness=12)  # , color=(255,0,0))
        # lines_unwarped, Minv = self.warp_image(dst_points, source_pts, blank_left | blank_right)

        # warp line (which was drawn on birdeye's view image) and project onto regular photo from car camera
        # l_line_unwarped, Minv = self.warp_image(dst_points, source_pts, blank_left)
        # r_line_unwarped, Minv = self.warp_image(dst_points, source_pts, blank_right)

        # overlay line over car camera image
        final_image = image.copy()
        final_image = cv2.addWeighted(final_image, 1, img_lines, .4, 0)
        # indices = lines_unwarped.nonzero()
        # final_image[indices[0], indices[1]] = line_color
        # cv2.imwrite(r'./output/street_w_lane_line_overlayed.jpg', final_image)

        if self.debug and self.frame_num % self.show_every_nth_frame == 0:
            self.display_two_ims(binary_image_warped, final_image, 'L', 'R')
            # cv2.imshow(self.windowName, final_image)
            # cv2.waitKey(0)

        # determine line curvature
        my = 30 / 720  # meters per pixel in y dimension
        mx = 3.7 / 800  # meters per pixel in x dimension
        left_coeffs_meters, right_coeffs_meters = self.get_poly_coeffs(leftx, lefty, rightx, righty, ym=my, xm=mx)
        self.left_curvature, self.right_curvature = self.measure_curvature(left_coeffs_meters, right_coeffs_meters, ploty, ym_per_pix=my)
        y = np.max(ploty)
        left_x = (left_coeffs[0] * (y ** 2)) + (left_coeffs[1] * y) + left_coeffs[2]
        right_x = (right_coeffs[0] * (y ** 2)) + (right_coeffs[1] * y) + right_coeffs[2]
        offset_pixels, self.offset_meters = self.find_offset_from_center(left_x, right_x, final_image.shape, mx)
        # print(offset_pixels, offset_meters)
        # print(left_curvature, right_curvature)
        return final_image


    def __call__(self, *args, **kwargs):
        if self.image is not None:
           frame = self.camera_obj.undistort_image(self.image, self.mtx, self.dist)
           self.find_lane_lines(frame)
        elif self.video is not None:
            cap = cv2.VideoCapture(self.video)

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1280, 720))
            while(1):
                ret, frame = cap.read()
                if not ret:
                    break
                else:
                    print("Processing frame {}".format(self.frame_num))
                    # if self.frame_num <= 960:
                    #     self.frame_num += 1
                    #     continue
                    # if self.frame_num > 1080:
                    #     break
                    frame = self.camera_obj.undistort_image(frame, self.mtx, self.dist)
                    out_frame = self.find_lane_lines(frame)
                    cv2.putText(out_frame, "Curvature of l and r lanes, respectively: {}, {} (m)".format("%.2f" % self.left_curvature, "%.2f" % self.right_curvature),
                    (50, 70), fontScale=.7, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0 ,0 ,0))
                    cv2.putText(out_frame,
                                "Car's distance from center of lane: {} (m)".format("%.2f" % self.offset_meters),
                                (50, 100), fontScale=.7, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0))
                    cv2.putText(out_frame,
                                "Frame {}, {}, {}".format(self.frame_num, "Poly Refitting" if self.poly_refit else "Sliding Boxes", self.conditions[2]),
                                (50, 130), fontScale=.7, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0))
                    if self.debug and self.frame_num % (self.show_every_nth_frame) == 0:
                        cv2.imshow("Test", out_frame)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    out.write(out_frame)
                    self.frame_num += 1
            cap.release()
            out.release()

            print("Lanes found by sliding boxes:", self.lanes_found_w_boxes)
            print("Lanes found by poly refitting:", self.lanes_found_w_poly_refit)
            print(self.conditions_counter)




if __name__ == '__main__':
    camera = Camera(images_folder=r'.\camera_cal')
    ret, mtx, dist, rvecs, tvecs = camera.find_camera_calib()

    # image = cv2.imread(r'./test_images/test1.jpg')
    # video = r'project_video.mp4'
    video = r'project_video.mp4'
    # Turn debug to 'True' to have an interactive experience of the algorithm
    lane_finder = LaneFinder(camera, video=video, debug=False)
    lane_finder()

