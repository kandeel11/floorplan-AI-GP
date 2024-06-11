# floorplan-AI-GP
Our project harnesses the capabilities of artificial intelligence and deep learning to automate the creation of detailed 2D and 3D floorplans, transforming the architectural design process. By leveraging advanced machine learning techniques, we aim to provide an efficient solution for architects, builders, and real estate professionals.
### README

# AI-Powered 2D and 3D Floorplan Generation

This project harnesses the capabilities of artificial intelligence and deep learning to automate the creation of detailed 2D and 3D floorplans, transforming the architectural design process. By leveraging advanced machine learning techniques, we aim to provide an efficient solution for architects, builders, and real estate professionals.

## Key Features

1. **Automated 2D Floorplan Generation:**
   - Utilizing convolutional neural networks (CNNs), our system converts basic sketches or raw data into precise 2D floorplans. These include detailed layouts of rooms, walls, windows, doors, and other architectural elements.

2. **3D Model Creation:**
   - Building on 2D floorplans, our AI system generates comprehensive 3D models. These models provide an immersive view, allowing users to visualize space dimensions, furniture placement, and architectural design in a virtual environment.

3. **User-Friendly Interface:**
   - A web-based or desktop application with an intuitive interface allows users to upload sketches or input data. Users can customize and adjust the generated floorplans and 3D models according to their specific needs.

4. **Real-Time Adjustments and Feedback:**
   - The system allows for real-time modifications and provides instant feedback, ensuring that the design meets the user’s requirements. Interactive tools enable fine-tuning of dimensions, layouts, and aesthetics.

5. **Integration with Existing Tools:**
   - Compatibility with popular architectural design software like AutoCAD, SketchUp, and Revit allows seamless integration into existing workflows. Export options in various formats, including DWG, DXF, and OBJ, facilitate easy sharing and collaboration.

## Applications

- **Architecture and Design:**
  - Simplifies the design process by quickly generating floorplans, assisting in remodeling and renovation projects.
- **Real Estate:**
  - Enhances property listings with 3D tours and interactive floorplans, providing potential buyers with a clear understanding of property layouts.
- **Construction:**
  - Streamlines project planning and management with precise floorplans, improving communication between builders, designers, and clients.

## Future Enhancements

- Incorporating augmented reality (AR) to visualize floorplans in actual environments.
- Enhancing AI models to support more complex designs and unconventional layouts.
- Developing collaborative tools for multiple users to work on the same project simultaneously.

## Installation

### Requirements

- Python 3.10
- Conda environment
- Torch
- CUDA
- PIL (Pillow)
- OpenCV (cv2)
- NumPy

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/kandeel11/floorplan-AI-GP.git
   cd floorplan-AI-GP
   ```

2. **Create and activate the Conda environment:**

   ```bash
   conda create --name floorplan-env python=3.10
   conda activate floorplan-env
   ```

3. **Install the required packages:**

   ```bash
   conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
   pip install pillow opencv-python-headless numpy
   ```

4. **Run the application:**
     download trainedmodel in drive link  "https://drive.google.com/file/d/1BnxJhd-XIv0BYnmjRM13Z1_igqYypZPt/view?usp=sharing"
     make folder "trained_model"
   ```bash
   python LinkAll.py
   ```

This project aims to revolutionize architectural design and visualization, making the process more efficient, accurate, and accessible through AI and deep learning.
