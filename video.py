from pathlib import Path

import cv2

from slam import DataFolder


def main():
    data = DataFolder.load(Path("data/machine_hall/MH_01_easy/mav0"))

    first_timestamp_ns = data.cam_timestamps_ns[0]
    for timestamp_ns in data.cam_timestamps_ns:
        img = cv2.imread(str(data.get_cam0_image_path(timestamp_ns)))
        time_s = (timestamp_ns - first_timestamp_ns) / 1e9
        cv2.putText(img, f"{time_s:.2f} s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("cam0", img)
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
