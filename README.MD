# Kolosal AI File Processing API 🤖

This project implements a Flask-based API that leverages the Kolosal AI platform to process various file types (images, PDFs, videos, and text) and generate intelligent responses. It serves as a backend for applications requiring AI-powered content analysis and generation. The API handles file uploads, extracts relevant information, interacts with the Kolosal AI API, and returns the AI-generated responses.

## 🚀 Key Features

- **File Upload and Processing:** Accepts file uploads of various types (text, PDF, images, videos).
- **Content Extraction:** Extracts text from PDF files and captures key frames from video files. Images are directly encoded.
- **AI Interaction:** Sends extracted content to the Kolosal AI API for analysis and processing.
- **Response Handling:** Receives and returns AI-generated responses in JSON format.
- **Error Handling:** Implements robust error handling for file uploads, processing, and API interactions.
- **Static File Serving:** Serves uploaded files from a designated directory.
- **CORS Enabled:** Allows requests from different origins, enabling seamless frontend integration.

## 🛠️ Tech Stack

| Category    | Technology           | Description                                                              |
|-------------|----------------------|--------------------------------------------------------------------------|
| **Backend** | Flask                | Python web framework for building the API.                               |
|             | Python               | Programming language used for the backend logic.                         |
| **AI**      | Kolosal AI API       | AI platform for content analysis and generation.                         |
|             | OpenAI Python Library | Interacting with the Kolosal AI API.                                    |
| **Data Processing** | OpenCV (cv2)         | Capturing frames from video files.                                     |
|             | PyPDF2               | Extracting text from PDF files.                                          |
|             | Pillow (PIL)         | Image processing.                                                        |
|             | base64               | Encoding images and video frames.                                        |
| **Utilities** | os                   | Interacting with the operating system.                                   |
|             | werkzeug.utils       | Secure filename handling.                                                |
|             | io                   | Handling in-memory byte streams.                                         |
|             | dotenv               | Loading environment variables from a `.env` file.                        |
| **Other**   | flask\_cors          | Enabling Cross-Origin Resource Sharing (CORS).                           |

## 📦 Getting Started

### Prerequisites

- Python 3.6+
- pip package manager
- An active Kolosal AI API key
- OpenCV (cv2)
    ```bash
    pip install opencv-python
    ```

### Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  Create a virtual environment (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4.  Set up environment variables:

    -   Create a `.env` file in the root directory.
    -   Add your Kolosal AI API key:

        ```
        KOLOSAL_API_KEY=YOUR_KOLOSAL_API_KEY
        ```

### Running Locally

1.  Run the Flask application:

    ```bash
    python app.py
    ```

2.  The API will be accessible at `http://127.0.0.1:5000`.

## 💻 Usage

The API provides the following endpoints:

-   **`POST /upload`**: Uploads a file for processing.  The file type is automatically detected and processed accordingly.  The API expects a `file` parameter in the request.

    Example (using `curl`):

    ```bash
    curl -X POST -F "file=@path/to/your/file.pdf" http://127.0.0.1:5000/upload
    ```

-   **`GET /<filename>`**: Serves static files from the `uploads` directory.

## 📂 Project Structure

```
├── app.py               # Main Flask application file
├── requirements.txt   # Project dependencies
├── uploads/             # Directory for storing uploaded files
├── venv/                # Virtual environment (optional)
├── .env                 # Environment variables (API key)
└── README.md            # Project documentation
```

## 📸 Screenshots



## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with descriptive messages.
4.  Submit a pull request.

## 📝 License

This project is licensed under the [MIT License](LICENSE).

## 📬 Contact

If you have any questions or issues, please feel free to contact me at [your_email@example.com](mailto:your_email@example.com).

## 💖 Thanks

Thank you for checking out this project! I hope it helps you build amazing AI-powered applications.

This is written by [readme.ai](https://readme-generator-phi.vercel.app/).
