### Program without User Input / only the image feeded in program will be subjected to image processing ###

import cv2
import numpy as np

class CheckerboardEdgeDetector:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.edges = self.detect_edges()

    def detect_edges(self):
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)
        # Finding edges using the Canny edge detector method
        edges = cv2.Canny(blurred, 200, 200)
        # Dilating the edges to connect broken edges
        dilated = cv2.dilate(edges, None, iterations=1)
        # Eroding the dilated edges to restore original size
        edges = cv2.erode(dilated, None, iterations=1)
        return edges

    def find_contours(self):
        #cv2.findCounters had been used to find the counters
        contours, _ = cv2.findContours(self.edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def draw_edges(self):
        edged_image = self.image.copy()
        #cv2.drawCountours had been used to draw the counters
        cv2.drawContours(edged_image, self.find_contours(), -1, (0, 255, 0), 2)
        return edged_image

    def display_edges(self):
        edged_image = self.draw_edges()
        cv2.imshow('Highlighted Edges', edged_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Since the images are in same folder Just provide the Image name for Edge Detection ## If not we should provide the complete path of image
    detector = CheckerboardEdgeDetector('Image_1.png')
    detector.display_edges()