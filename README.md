# NoNoFilter

**NoNoFilter** is a high-performance, local application designed to review and censor profanity in audiobooks, videos, and other media files. By leveraging state-of-the-art AI models, it allows you to automatically detect, review, and "bleep" or mute specific terms while keeping all your data strictly on your own machine.

## Key Features

- **Local & Private**: No audio or transcripts ever leave your computer. All AI analysis is performed locally.
- **Advanced AI Models**:
  - **Faster-Whisper**: Optimized OpenAI Whisper models for excellent accuracy and speed.
  - **NVIDIA Parakeet TDT**: Ultra-fast, multilingual transcription with precision word-level timestamps.
- **Smart Review System**:
  - **Censor UI**: Automatically masks vowels (e.g., `sh*t`) in the interface for a clean review experience.
  - **Instant Preview**: Listen to original vs. censored audio side-by-side before exporting.
  - **Precise Timing**: Razor-sharp bleeping that targets specific words without cutting off clean speech.
- **Robust Engine**:
  - **Resume Support**: Smartly picks up from where you left off if a session is interrupted.
  - **Atomic Saving**: Prevents data corruption by ensuring project files are saved safely.
  - **Format Support**: Handles MP3, M4B, M4A, MP4, MKV, and more.

## Installation

### Prerequisites

1. **Python 3.12 (Recommended)**: While newer versions may work, Python 3.12 is the most stable and tested version for our AI dependencies.
2. **FFmpeg**: Required for audio processing. Ensure it is installed and added to your system `PATH`.

### Setup

1. **Clone the repository**:
   ```powershell
   git clone https://github.com/eqwreqwrwr/NoNoFilter.git
   cd NoNoFilter
   ```

2. **Install dependencies**:
   ```powershell
   py -3.12 -m pip install .
   ```

3. **GPU Acceleration (Recommended)**:
   If you have an NVIDIA GPU, run the included script to install the high-performance CUDA version of PyTorch:
   ```powershell
   py -3.12 install_cuda_support.py
   ```

## Troubleshooting

- **Installation Hangs**: If the installation gets stuck at "Building wheel for onnx", ensure you are using Python 3.12. Newer versions of Python may force a slow source-compilation of AI libraries.
- **FFmpeg Errors**: If you see "FFmpeg not found", download it from [ffmpeg.org](https://ffmpeg.org/) and ensure the `bin` folder is in your Windows environment variables.
- **UI Scaling**: If the app window looks blurry, right-click the Python shortcut/executable, go to Properties > Compatibility > Change high DPI settings, and override high DPI scaling.

## Usage

Launch the application using the following command:

```powershell
$env:PYTHONPATH="src"; py -3.12 -m nono_filter.main gui
```

1. **Import**: Select your audiobook or video file.
2. **Transcribe**: Choose a model (e.g., `turbo` or `parakeet`) and start the local AI analysis.
3. **Match**: Add terms to your profanity list and click "Matches" to find them in the transcript.
4. **Review**: Listen to previews and approve the bleeps.
5. **Export**: Render the final censored media file.

## Acknowledgments & Attribution

NoNoFilter is built upon the following incredible open-source projects:

- **NVIDIA Parakeet Models**: Released by NVIDIA under **CC BY 4.0**.
- **Faster-Whisper**: High-performance implementation of OpenAI's Whisper model by SYSTRAN.
- **NVIDIA NeMo**: Conversational AI toolkit by NVIDIA.
- **PySide6**: The official Python modules from the Qt project.
- **FFmpeg**: The universal multimedia framework.

## License

NoNoFilter is released under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** license.

- **Non-Commercial**: You may not use this material for commercial purposes.
- **Attribution**: You must give appropriate credit to the original author.
- **ShareAlike**: If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.
