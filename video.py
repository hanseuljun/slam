from pathlib import Path

import cv2

from slam import DataFolder


def main():
    data = DataFolder.load(Path("data/machine_hall/MH_01_easy/mav0"))

    for timestamp_ns in data.cam_timestamps_ns:
        img = cv2.imread(str(data.get_cam0_image_path(timestamp_ns)))
        cv2.imshow("cam0", img)
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
