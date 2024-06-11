

data = {"mask":[[44,386],[44,664],[682,664],[682,524],[962,524],[962,174],[682,174],[682,314],[537,314],[537,34],[44,34]],"door_pos":[826,174],"area":71.16000000000003}
data = {
    "mask": [[140, 126], [140, 224], [249, 224], [249, 168], [247, 168], [247, 70], [217, 70], [217, 140], [175, 140], [175, 126]],
    "door_pos": [233, 70],
    "area": 117.5
}
data = {"mask":[[209,701],[560,701],[560,560],[630,560],[630,701],[1120,701],[1120,420],[910,420],[910,350],[1082,350],[1082,200],[1153,200],[1153,97],[1073,97],[1073,13],[747,13],[747,70],[478,70],[478,200],[140,200],[140,350],[350,350],[350,490],[209,490]],"door_pos":[350,425],"area":81.38000000000004}
# Extract mask coordinates


# Check if door_pos[1] is in the list of y-coordinates in the mask

def CreateMask(data):
    import cv2
    import numpy as np
    from PIL import Image
    import os
    import shutil
    mask_coordinates = np.array(data["mask"])

    # Find max value for x and y coordinates separately
    max_values = np.max(mask_coordinates, axis=0)

    # Find max value for "door_pos"
    max_value_door_pos = np.max(data["door_pos"])

    # Get the overall max value by combining mask and door_pos max values
    overall_max_value = max(max_values[0], max_values[1], max_value_door_pos) + 50

    # Normalize mask coordinates
    normalized_mask = (mask_coordinates / overall_max_value * 256).astype(int).tolist()

    # Normalize door_pos coordinates
    normalized_door_pos = ((np.array(data["door_pos"]) / overall_max_value) * 256).astype(int).tolist()

    # Normalize area
    normalized_area = (data["area"] / overall_max_value ** 2 * 256).tolist()

    # Update the data with normalized values
    data = {
        "mask": normalized_mask,
        "door_pos": normalized_door_pos,
        "area": normalized_area
    }

    # Target dimensions
    height, width = 256, 256
    # Create a blank image with an alpha channel
    image = np.zeros((height, width, 4), dtype=np.uint8)  # 4 channels (BGR + alpha)
    # Convert the mask points to NumPy array
    boundary_mask_pts = np.array([data["mask"]], dtype=np.int32)
    # Create a mask for the boundary
    boundary_mask = np.zeros((256, 256), dtype=np.uint8)
    contours = np.array([data["mask"]], dtype=np.int32)
    cv2.drawContours(boundary_mask, contours, 0, 127,2)
    x_coordinates = [point[0] for point in data["mask"]]
    door_x_coordinate_in_mask = data["door_pos"][0] in x_coordinates
    if (door_x_coordinate_in_mask):
        cv2.line(boundary_mask, (int(data["door_pos"][0]),int(data["door_pos"][1])-10), (int(data["door_pos"][0]),int(data["door_pos"][1])+10), 255, 2)
    y_coordinates = [point[1] for point in data["mask"]]
    door_y_coordinate_in_mask = data["door_pos"][1] in y_coordinates
    if(door_y_coordinate_in_mask):
        cv2.line(boundary_mask, (int(data["door_pos"][0])-10,int(data["door_pos"][1])), (int(data["door_pos"][0])+10,int(data["door_pos"][1])), 255, 2)
    boundary_mask_colored = cv2.cvtColor(boundary_mask, cv2.COLOR_GRAY2BGRA)
    # Create a mask for the inside region
    inside_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(inside_mask, [boundary_mask_pts], 255)
    # Find the bounding box of the inside mask
    y, x = np.where(inside_mask == 255)
    top, left = np.min(y), np.min(x)
    bottom, right = np.max(y), np.max(x)
    # Calculate the center of the bounding box
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    # Calculate the offset to center the image
    offset_x = (width // 2) - center_x
    offset_y = (height // 2) - center_y
    # Apply the offset to the masks
    boundary_mask = cv2.warpAffine(boundary_mask, np.float32([[1, 0, offset_x], [0, 1, offset_y]]), (width, height))
    inside_mask = cv2.warpAffine(inside_mask, np.float32([[1, 0, offset_x], [0, 1, offset_y]]), (width, height))
    # Apply the inside mask to the alpha channel
    image[:, :, 3] = inside_mask
    image[:, :, 0] = boundary_mask
    # Draw the boundary mask in layer 1 (Green channel)
    output_dir = 'befor_Ai'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    output = Image.fromarray(image)
    output.save(f'{output_dir}/result_befor.png')
    return output