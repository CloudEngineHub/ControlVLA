import cv2
import threading
import numpy as np
import json
from real_world.time import Rate
from umi.common.usb_util import reset_all_elgato_devices
from umi.common.cv_util import parse_fisheye_intrinsics, FisheyeRectConverter
from diffusion_policy.common.cv2_util import get_image_transform, optimal_row_cols

class CameraHandler:
    def __init__(self, camera_id,
                 calibration_file='./example/calibration/gopro_intrinsics_2_7k.json'):
        # Reset devices
        reset_all_elgato_devices()

        # Initialize camera
        self.camera = cv2.VideoCapture(camera_id)
        w, h = (1920, 1080)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera.set(cv2.CAP_PROP_FPS, 60)

        # Load camera calibration parameters and construct fisheye rectifier
        with open(calibration_file, 'r') as f:
            intr_dict = json.load(f)
        opencv_intr_dict = parse_fisheye_intrinsics(intr_dict)
        self.fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=(224, 224),
            out_fov=85
        )

        # Initialize latest image buffer and its lock
        self.latest_image = None
        self.latest_image_lock = threading.Lock()

        # Thread control event
        self.shutdown_event = threading.Event()

        # Thread handles
        self.capture_thread = None
        self.display_thread = None

    def _transform(self, image):
        """
        Image preprocessing: apply perspective transform, then fisheye rectification.
        """
        f = get_image_transform(
            input_res=(1920, 1080),
            output_res=(2704, 2028)
        )
        img_transformed = f(image)
        img_rect = self.fisheye_converter.forward(img_transformed)
        return img_rect

    def _capture_loop(self):
        """
        120 Hz capture thread: continuously read frames, process, and update latest_image.
        """
        rate = Rate(120)
        while not self.shutdown_event.is_set():
            ret, frame = self.camera.read()
            if not ret:
                rate.sleep()
                continue
            image = np.array(frame)
            processed = self._transform(image)
            with self.latest_image_lock:
                self.latest_image = processed[:, :, ::-1]
            rate.sleep()

    def _display_loop(self):
        """
        30 Hz display thread: fetch image from latest_image, display, and listen for 'q' to exit.
        """
        rate = Rate(30)
        while not self.shutdown_event.is_set():
            with self.latest_image_lock:
                disp_img = self.latest_image.copy() if self.latest_image is not None else None
            if disp_img is not None:
                cv2.imshow('preview', disp_img[:, :, ::-1])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.shutdown_event.set()
                break
            rate.sleep()

    def start(self):
        """
        Start capture and display threads.
        """
        if not self.camera.isOpened():
            raise RuntimeError("摄像头未打开")
        self.shutdown_event.clear()
        self.capture_thread = threading.Thread(target=self._capture_loop, name="CaptureThread")
        self.display_thread = threading.Thread(target=self._display_loop, name="DisplayThread")
        self.capture_thread.start()
        self.display_thread.start()

    def get_latest_image(self):
        """
        Public interface: return a copy of the latest image.
        """
        with self.latest_image_lock:
            return None if self.latest_image is None else self.latest_image.copy()

    def stop(self):
        """
        Stop all threads and release camera and window resources.
        """
        self.shutdown_event.set()

        # Wait for threads to exit
        if self.capture_thread is not None:
            self.capture_thread.join()
            self.capture_thread = None
        if self.display_thread is not None:
            self.display_thread.join()
            self.display_thread = None

        # Release camera and window resources
        if self.camera.isOpened():
            self.camera.release()
        cv2.destroyAllWindows()

    def __del__(self):
        """
        Ensure all threads and resources are closed when the instance is destroyed.
        """
        try:
            self.stop()
        except Exception:
            pass

# Usage example
if __name__ == "__main__":
    from umi.common.usb_util import get_sorted_v4l_paths
    v4l_paths = get_sorted_v4l_paths(by_id=False)
    print(v4l_paths)
    v4l_path = v4l_paths[0]
    cam_handler = CameraHandler(v4l_path)
    cam_handler.start()
    try:
        # The main loop can periodically call get_latest_image() to process images
        rate = Rate(30)
        while not cam_handler.shutdown_event.is_set():
            img = cam_handler.get_latest_image()
            if img is not None:
                print("Latest image mean: {:.2f}".format(np.mean(img)))
            rate.sleep()  # 30 Hz main loop
    except KeyboardInterrupt:
        pass
    finally:
        cam_handler.stop()
