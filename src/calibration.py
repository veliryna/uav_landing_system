# run once to calibrate camera and update info in config.yaml
import cv2, numpy as np, glob

# no calibration dataset yet
images = glob.glob('calib_dataset/*.jpg')
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

obj_pts, img_pts = [], []
for fname in images:
    gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6))
    if ret:
        obj_pts.append(objp)
        img_pts.append(corners)

ret, mtx, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, gray.shape[::-1], None, None)
print(f"fx={mtx[0,0]:.2f} fy={mtx[1,1]:.2f} cx={mtx[0,2]:.2f} cy={mtx[1,2]:.2f}")
print(f"dist={dist.flatten().tolist()}")