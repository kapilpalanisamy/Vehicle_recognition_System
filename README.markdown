# Vehicle Recognition System - Project Starter Guide

This project implements a Vehicle Recognition System that processes video input to detect vehicles, track their speeds, recognize license plates, and check against a blacklist. This README provides instructions on how to set up and run the project.

## Prerequisites

Before running the project, ensure you have the following installed:

- **Python 3.8 or higher**: Download and install from [python.org](https://www.python.org/downloads/).
- **pip**: Python package manager (usually included with Python).
- **Virtual Environment** (recommended): To isolate dependencies.
- **CUDA-enabled GPU** (optional but recommended): For faster processing with GPU acceleration.
- **FFmpeg**: Required for video processing with OpenCV. Install it:
  - Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html) or install via Chocolatey (`choco install ffmpeg`).
  - Linux: `sudo apt-get install ffmpeg`
  - macOS: `brew install ffmpeg`

## Setup Instructions

1. **Clone the Repository or Download the Project Files**

   If using Git, clone the repository:
   ```bash
   git clone <repository-url>
   cd vehicle-recognition-system
   ```

   Alternatively, download and extract the project files to a folder.

2. **Create a Virtual Environment**

   Create and activate a virtual environment to manage dependencies:
   ```bash
   python -m venv venv
   ```

   Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/macOS: `source venv/bin/activate`

3. **Install Dependencies**

   Install the required Python packages using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

   If there’s no `requirements.txt`, install the following packages manually:
   ```bash
   pip install opencv-python numpy torch torchvision ultralytics supervision easyocr pandas xlsxwriter
   ```

   For GPU support, ensure you have the correct version of PyTorch with CUDA:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

   Verify the installations by running:
   ```bash
   python -c "import cv2, numpy, torch, easyocr, ultralytics, supervision, pandas"
   ```

4. **Download YOLOv8 Model Weights**

   The project uses YOLOv8 (`yolov8n.pt`) for vehicle and license plate detection. The Ultralytics library automatically downloads the model weights when needed, but you can manually download them from the [Ultralytics YOLOv8 GitHub](https://github.com/ultralytics/assets/releases) and place them in the project directory if preferred.

5. **Prepare Input Video**

   The system processes a video file specified in the `INPUT_VIDEO` variable. The default is:
   ```
   ip/videos/ACCIDENT Happened While StreeT Racing _ Duke 200 vs R15 V3 _ Extreme Traffic Filter.mp4
   ```

   Ensure the video file exists at the specified path or update the `INPUT_VIDEO` variable in the script to point to your video file. For example:
   ```python
   INPUT_VIDEO = "path/to/your/video.mp4"
   ```

6. **Configure Blacklist**

   The `BLACKLIST` variable contains license plates to flag as alerts. Modify it in the script as needed:
   ```python
   BLACKLIST = ["KA01MJ2022", "MH02AB1234", "DL01CD5678", "TN07EF9012", "AP02CA1600"]
   ```

## Running the Project

1. **Run the Script**

   With the virtual environment activated, run the Python script:
   ```bash
   python vehicle_recognition_system.py
   ```

   Replace `vehicle_recognition_system.py` with the actual name of your script file.

2. **Interact with the System**

   Once running, the system will:
   - Process the video and display the output in a window.
   - Detect vehicles, track their speeds (if enabled), and recognize license plates (if enabled).
   - Flag blacklisted vehicles with an alert sound and red bounding boxes.
   - Save logs to an Excel file (`vehicle_log.xlsx`) in the same directory as the input video.

   **Key Controls**:
   - `s`: Toggle speed detection (default: OFF for better performance).
   - `o`: Toggle license plate OCR (default: OFF for better performance).
   - `+`: Increase font size for displayed text.
   - `-`: Decrease font size for displayed text.
   - `q`: Quit the application.

3. **Output**

   - **Video Output**: Real-time display with bounding boxes, vehicle IDs, speeds, license plates, and blacklist alerts.
   - **Excel Log**: Vehicle data (timestamp, ID, type, license plate, speed, blacklist status) saved to `vehicle_log.xlsx`.

## Troubleshooting

- **Video Not Found**: Ensure the `INPUT_VIDEO` path is correct and the file is accessible.
- **Module Not Found**: Verify all dependencies are installed. Re-run `pip install` commands.
- **Low FPS**: Enable frame skipping or disable speed/OCR modes (`s` or `o` keys). Use a GPU for better performance.
- **CUDA Errors**: Ensure PyTorch with CUDA is installed correctly, and your GPU drivers are up to date. If issues persist, the script will fall back to CPU.
- **Excel Save Issues**: Check write permissions in the output directory and ensure `xlsxwriter` is installed.

## Notes

- **Performance**: For faster processing, use a CUDA-enabled GPU, lower the input video resolution, or enable frame skipping.
- **Accuracy**: The system uses YOLOv8n with estimated accuracies (vehicle detection: ~89%, plate detection: ~75%, OCR: 60-85%). Accuracy depends on video quality and lighting.
- **Customization**: Adjust `speed_threshold`, `ocr_frame_skip`, or `ppm` (pixels per meter) in the script for your use case.
- **Windows Sound**: The blacklist alert uses `winsound`. If it fails, a warning will be printed, but the system will continue running.

## Additional Resources

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [EasyOCR Documentation](https://www.jaided.ai/easyocr/documentation/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

For further assistance, check the project’s source code comments or contact the repository maintainer.