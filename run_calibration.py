import cv2
import numpy as np
import glob
import os

def run_stereo_calibration():
    CHESS_ROWS = 6
    CHESS_COLS = 9
    SQUARE_SIZE = 42.0  # MEASURE YOUR SQUARES and update this (mm)

    objp = np.zeros((CHESS_ROWS * CHESS_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESS_COLS, 0:CHESS_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    obj_points = []
    img_points0 = []
    img_points1 = []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    images0 = sorted(glob.glob("data/calib_cam0/*.png"))
    images1 = sorted(glob.glob("data/calib_cam1/*.png"))

    if len(images0) == 0:
        print("No calibration images found!")
        return

    print(f"Found {len(images0)} image pairs. Processing...")

    img_size0 = None
    img_size1 = None

    for img_file0, img_file1 in zip(images0, images1):
        img0 = cv2.imread(img_file0)
        img1 = cv2.imread(img_file1)
        gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        if img_size0 is None:
            img_size0 = gray0.shape[::-1]  # (width, height)
            img_size1 = gray1.shape[::-1]
            print(f"Camera 0 resolution: {img_size0}")
            print(f"Camera 1 resolution: {img_size1}")

        found0, corners0 = cv2.findChessboardCorners(gray0, (CHESS_COLS, CHESS_ROWS), None)
        found1, corners1 = cv2.findChessboardCorners(gray1, (CHESS_COLS, CHESS_ROWS), None)

        if found0 and found1:
            corners0 = cv2.cornerSubPix(gray0, corners0, (11, 11), (-1, -1), criteria)
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points0.append(corners0)
            img_points1.append(corners1)
        else:
            print(f"Skipping {os.path.basename(img_file0)} - corners not found in both")

    print(f"\nUsing {len(obj_points)} valid pairs for calibration...")

    if len(obj_points) < 10:
        print("WARNING: Less than 10 valid pairs. Try to capture more!")

    # Calibrate each camera individually (handles different resolutions)
    print("Calibrating camera 0...")
    ret0, K0, dist0, _, _ = cv2.calibrateCamera(obj_points, img_points0, img_size0, None, None)
    print(f"Camera 0 reprojection error: {ret0:.4f}")

    print("Calibrating camera 1...")
    ret1, K1, dist1, _, _ = cv2.calibrateCamera(obj_points, img_points1, img_size1, None, None)
    print(f"Camera 1 reprojection error: {ret1:.4f}")

    # Stereo calibration
    # If cameras have different resolutions, we need to resize cam1 images
    # to match cam0 for stereo calibration, OR normalise the points
    
    # Scale cam1 points if resolutions differ
    if img_size0 != img_size1:
        print(f"\nDifferent resolutions detected. Normalising cam1 points...")
        scale_x = img_size0[0] / img_size1[0]
        scale_y = img_size0[1] / img_size1[1]
        
        # Scale K1 to match cam0 image size
        K1[0, 0] *= scale_x  # fx
        K1[1, 1] *= scale_y  # fy
        K1[0, 2] *= scale_x  # cx
        K1[1, 2] *= scale_y  # cy
        
        # Scale the 2D points
        img_points1_scaled = []
        for pts in img_points1:
            scaled = pts.copy()
            scaled[:, :, 0] *= scale_x
            scaled[:, :, 1] *= scale_y
            img_points1_scaled.append(scaled)
        img_points1 = img_points1_scaled

    print("Running stereo calibration...")
    flags = cv2.CALIB_FIX_INTRINSIC
    ret_stereo, K0, dist0, K1, dist1, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points0, img_points1,
        K0, dist0, K1, dist1,
        img_size0, criteria=criteria, flags=flags
    )
    print(f"\nStereo reprojection error: {ret_stereo:.4f}")

    if ret_stereo < 1.0:
        print("EXCELLENT calibration!")
    elif ret_stereo < 2.0:
        print("GOOD calibration.")
    else:
        print("WARNING: High error. Recapture calibration images.")

    # Projection matrices
    P0 = K0 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = K1 @ np.hstack((R, T))

    # Save
    os.makedirs("data", exist_ok=True)
    np.savez("data/stereo_calib.npz",
             K0=K0, dist0=dist0,
             K1=K1, dist1=dist1,
             R=R, T=T, E=E, F=F,
             P0=P0, P1=P1,
             img_size0=np.array(img_size0),
             img_size1=np.array(img_size1))

    print("\nCalibration saved to data/stereo_calib.npz")
    print(f"\nRotation between cameras:\n{R}")
    print(f"\nTranslation between cameras:\n{T.flatten()}")

if __name__ == "__main__":
    run_stereo_calibration()