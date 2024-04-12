import cv2
import os

class CheckerboardEdgeDetector:
    def __init__(self, image):
        self.image = image
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.edges = self.detect_edges()

    def detect_edges(self):
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)
        # Finding edges using the Canny edge detector
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

def process_images():
    # Get the list of image files in the current directory
    image_files = [file for file in os.listdir() if file.endswith('.jpg') or file.endswith('.png')]

    # Filter out the processed images
    image_files = [file for file in image_files if not file.startswith('processed_')]

    # Process each image
    for idx, image_file in enumerate(image_files, start=1):
        # Read the image
        image = cv2.imread(image_file)
        if image is None:
            print("Error: Could not read image '" + image_file + "'. Skipping...")
            continue

        # Process the image
        detector = CheckerboardEdgeDetector(image)
        edged_image = detector.draw_edges()

        # Generate unique output filename
        output_file = "processed_img_" + str(idx) + ".jpg"
        
        # Save the processed image
        cv2.imwrite(output_file, edged_image)
        
        print("Processed image saved as '" + output_file + "'")

if __name__ == "__main__":
    process_images()