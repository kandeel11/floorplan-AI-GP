# floorplan-AI-GP
Our project harnesses the capabilities of artificial intelligence and deep learning to automate the creation of detailed 2D and 3D floorplans, transforming the architectural design process. By leveraging advanced machine learning techniques, we aim to provide an efficient solution for architects, builders, and real estate professionals.
### README
![image](https://github.com/kandeel11/floorplan-AI-GP/assets/113936202/32f32dfc-b838-462d-a412-5061f6a35148)

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
## Models
![image](https://github.com/kandeel11/floorplan-AI-GP/assets/113936202/dc4312e1-4674-49ff-83de-927a9f605afb)
![image](https://github.com/kandeel11/floorplan-AI-GP/assets/113936202/92181a55-2e14-4902-b457-b21585e388b7)
![image](https://github.com/kandeel11/floorplan-AI-GP/assets/113936202/ce7c8d84-fbd2-4b84-bd73-7a7798dc262a)
![image](https://github.com/kandeel11/floorplan-AI-GP/assets/113936202/45340f2a-a2c2-4e36-9230-fd20afbf4eaf)
## output 
![0in10309](https://github.com/kandeel11/floorplan-AI-GP/assets/113936202/dc63c243-0023-4a6a-ae79-82b77d602d2f)![result](https://github.com/kandeel11/floorplan-AI-GP/assets/113936202/b27d846d-7aa6-4f23-a4a5-dcf73130aeed)![0in13478](https://github.com/kandeel11/floorplan-AI-GP/assets/113936202/5c91ce02-a439-404d-b251-7f42f46e91d3)![0in1382](https://github.com/kandeel11/floorplan-AI-GP/assets/113936202/7c187373-45b9-4027-8c1f-79d31f3febb0)![0in14764](https://github.com/kandeel11/floorplan-AI-GP/assets/113936202/7c3f1da0-5701-4fa1-8426-c8a79a17cb6a)![0in10581](https://github.com/kandeel11/floorplan-AI-GP/assets/113936202/065dda6c-7d04-48ac-9cbd-ab604a581160)


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
