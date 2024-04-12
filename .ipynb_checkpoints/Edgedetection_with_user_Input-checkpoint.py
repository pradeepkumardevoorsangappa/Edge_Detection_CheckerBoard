### Program with User Input / user can choose the image to be detect edges ###


import cv2
import numpy as np

class CheckerboardEdgeDetector:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError("Could not load image: " + image_path)
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

def select_image():
    while True:
        ## as there is image 1-4 with .png format and image 5 is with .jpg format
        try:
            image_number = int(input("Enter the image number (1-5): "))
            if image_number in range(1, 6):
                # select_image returns .jpg is number 5 is given as user input
                if image_number == 5:
                    return "Image_5.jpg"
                else:
                    # select_image returns .png is number other than 5 is given as user input
                    return "Image_" + str(image_number) + ".png"
            else:
                print("Invalid input. Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    try:
        image_path = select_image()
        detector = CheckerboardEdgeDetector(image_path)
        detector.display_edges()
        # Since try is used the program need 'except' or 'finally' block
    except FileNotFoundError as e:
        print(e)
