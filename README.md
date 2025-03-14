# Personalized-Doggy-Door
🐶Personalized Doggy Door– A CNN-powered smart doggy door that distinguishes between your personal dog and other dogs, ensuring only your pet gets access! 🚪✨ #DeepLearning #PyTorch #ComputerVision

## Features

- **Accurate Dog Identification:** Utilizes a Convolutional Neural Network (CNN) to identify and differentiate your dog from others.
- **Automated Access Control:** Integrates with a smart door mechanism to allow only your pet entry.
- **Efficient Data Handling:** Employs PyTorch’s DataLoader with custom dataset classes for robust image processing.
- **Real-World Application:** Blends deep learning with practical hardware integration.

## Key Libraries & Technologies

- **[PyTorch](https://pytorch.org/):** For building and training the CNN model.
- **[Torchvision](https://pytorch.org/vision/stable/):** For image transformations and processing.
- **[Pillow (PIL)](https://python-pillow.org/):** For image loading and preprocessing.
- **glob:** For efficient file path matching.
- **Jupyter Notebook:** For interactive development and experimentation.
- **Git & GitHub:** For version control and collaboration.

## Skills Demonstrated

- **Deep Learning & CNNs:** Designing, training, and optimizing convolutional neural networks for image classification.
- **Data Preprocessing:** Efficiently handling and transforming image data using Python libraries.
- **Computer Vision:** Implementing state-of-the-art image processing techniques.
- **Problem Solving:** Integrating software with hardware to solve a real-world problem.
- **Version Control:** Managing projects effectively with Git and GitHub.

## Project Structure

Personalized-Doggy-Door/ ├── data/ │ └── presidential_doggy_door/ │ ├── train/ │ │ ├── bo/ │ │ └── not_bo/ │ └── valid/ ├── models/ │ └── (trained model files) ├── notebooks/ │ └── Personalized Doggy Door.ipynb ├── README.md └── requirements.txt

markdown
Copy
Edit

*Note: Adjust the folder structure if necessary.*

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/daksharora2005/Personalized-Doggy-Door.git
Navigate to the Project Folder:
bash
Copy
Edit
cd Personalized-Doggy-Door
Install Dependencies:
bash
Copy
Edit
pip install -r requirements.txt
Ensure you have Python 3.x installed.
Usage
Dataset Setup:

Place your images in data/presidential_doggy_door/ under the respective train and valid folders.
Organize images into subfolders named bo (for your personal dog) and not_bo (for other dogs).
Run the Jupyter Notebook:

bash
Copy
Edit
jupyter notebook notebooks/Personalized\ Doggy\ Door.ipynb
Follow the Notebook Instructions:

Train and evaluate the CNN model.
Experiment with different parameters and monitor the performance.
Acknowledgements
Thanks to the open-source community for the amazing libraries and tools that made this project possible.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Happy coding and enjoy building your smart doggy door! 🚀🐶
