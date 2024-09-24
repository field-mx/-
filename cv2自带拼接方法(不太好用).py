import cv2
from main import resize_image

def main(image_paths):
    images = [cv2.imread(path) for path in image_paths]

    stitcher = cv2.Stitcher_create()
    status, stitched = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        cv2.imwrite('stitched_image(auto).jpg', stitched)
        cv2.imshow('Stitched Image', stitched)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Stitching failed with status {status}")




if __name__ == '__main__':
    image_paths = ['room1.jpg', 'room2.jpg', 'room3.jpg', 'room4.jpg']
    main(image_paths)
