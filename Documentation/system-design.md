# System Design Document

## 1. General Information

### 1.1 Version Control

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| April 08, 2025 | 1.0 | Initial system design document | Project Team |

### 1.2 Information Details

| Informational Item | Information |
|-------------------|-------------|
| Project Name | Image Deblurring System |
| Project Phase | Phase 1 |
| Project Iteration | 1 |

## 2. Class diagram description

### 2.1 CRC Template

The following CRC (Class-Responsibility-Collaborator) cards represent the main components of our Image Deblurring System:

| Class Name | Class Type | Characteristics |
|------------|------------|----------------|
| WebInterface | Boundary | User-facing component that handles image upload/download |
| ImageProcessor | Service | Manages image validation, preprocessing and postprocessing |
| DeblurModel | Service | Wraps the neural network for deblurring |
| Generator | Domain | Neural network model with dense field architecture |
| DenseBlock | Domain | Building block of the neural network |
| ResultManager | Service | Manages processed image storage and retrieval |
| User | External | User of the system |

### 2.2 CRC Cards Overview

#### WebInterface
**Responsibilities:**
- Accept image uploads from users
- Display deblurring results with side-by-side comparison
- Provide download functionality for processed images
- Show progress indicators during processing
- Provide intuitive user experience

**Collaborators:**
- ImageProcessor
- ResultManager
- User

#### ImageProcessor
**Responsibilities:**
- Validate uploaded image format and size
- Preprocess images for the neural network
- Optimize output images after processing
- Handle image conversion between formats

**Collaborators:**
- DeblurModel
- WebInterface
- ResultManager

#### DeblurModel
**Responsibilities:**
- Manage the deblurring neural network
- Execute the deblurring process
- Provide status updates on processing
- Handle GPU/CPU resource allocation

**Collaborators:**
- Generator
- ImageProcessor

#### Generator
**Responsibilities:**
- Implement the dense field neural network architecture
- Process image through dense blocks and skip connections
- Transform blurry input to clear output
- Maintain state during processing

**Collaborators:**
- DenseBlock
- DeblurModel

#### DenseBlock
**Responsibilities:**
- Process features at different scales
- Implement the connections between layers
- Apply convolutions with appropriate dilation factors
- Maintain batch normalization and activation functions

**Collaborators:**
- Generator

#### ResultManager
**Responsibilities:**
- Store processed images temporarily
- Provide access to results via unique identifiers
- Manage result expiration and cleanup
- Ensure secure access to results

**Collaborators:**
- WebInterface
- ImageProcessor

#### User
**Responsibilities:**
- Upload images for deblurring
- View deblurring results
- Download processed images

**Collaborators:**
- WebInterface

### 2.3 Class Diagram

```
+------------------+      +------------------+      +------------------+
|   WebInterface   |<---->|  ImageProcessor  |<---->|   DeblurModel    |
+------------------+      +------------------+      +------------------+
        ^                         |                         |
        |                         v                         v
        |                +------------------+      +------------------+
        |                |  ResultManager   |      |    Generator     |
        |                +------------------+      +------------------+
        |                                                  |
        v                                                  v
+------------------+                              +------------------+
|       User       |                              |    DenseBlock    |
+------------------+                              +------------------+
```

## 3. Component Architecture

### 3.1 Component Overview

The Image Deblurring System is composed of the following major components:

1. **Front-end Web Application**
    - User interface for image upload and result display
    - Built with responsive web design for various devices
    - Implements progress indicators and image comparison features

2. **Back-end API Service**
    - Handles requests from the web interface
    - Manages the processing queue
    - Coordinates the deblurring workflow

3. **Deblurring Engine**
    - Dense Field neural network implementation (PyTorch)
    - Pre and post-processing modules
    - Resource management for efficient processing

4. **Image Storage Service**
    - Temporary storage for input and result images
    - Secure access mechanisms
    - Automatic cleanup of expired data

### 3.2 Component Interactions

The components interact through well-defined interfaces:

```
+------------------+      +------------------+      +------------------+
|   Web Frontend   |<---->|  Backend API     |<---->|    Deblurring    |
|   (React/HTML)   |      |  (REST Service)  |      |    Engine (ML)   |
+------------------+      +------------------+      +------------------+
                                   ^
                                   |
                                   v
                           +------------------+
                           |  Image Storage   |
                           |     Service      |
                           +------------------+
```

### 3.3 Neural Network Architecture

The deblurring engine implements a Generator model with dense connections and dilated convolutions:

1. **Generator Structure**
    - Input Layer: Accepts RGB images (3 channels)
    - Head: Initial convolution layer (3x3)
    - Dense Field: Series of interconnected DenseBlocks with varying dilation factors
    - Skip Connections: Global and local skip connections to preserve image details
    - Output Layer: Final convolution with tanh activation

2. **DenseBlock Architecture**
    - Each DenseBlock contains two convolution layers
    - First layer: 1x1 convolution for channel reduction
    - Second layer: 3x3 convolution with dilation for multi-scale feature extraction
    - Batch normalization and LeakyReLU activations
    - Dropout for regularization (0.5 rate)

3. **Skip Connection Strategy**
    - Each DenseBlock output is concatenated with previous features
    - Global skip connection connects input directly to the final layers
    - This architecture ensures preservation of high-frequency details

## 4. Data Flow

### 4.1 Image Upload Flow

1. User uploads image through web interface
2. Frontend validates basic image parameters (format, size)
3. Image is sent to backend API
4. Backend performs additional validation
5. Image is stored temporarily with a unique identifier
6. Processing job is created and queued
7. User receives job identifier for tracking

### 4.2 Image Processing Flow

1. Processing worker picks job from queue
2. Image is loaded from storage
3. Image is preprocessed (resizing, normalization)
4. Preprocessed image is fed to the Generator model
5. Model processes the image through the Dense Field architecture
6. Output is post-processed (denormalization, format conversion)
7. Result is stored with reference to the original job
8. Job status is updated to completed

### 4.3 Result Retrieval Flow

1. Frontend polls backend for job status using job identifier
2. When job is complete, result image URL is returned
3. Frontend displays original and processed images side by side
4. User can zoom, pan, and compare the images
5. User can download the processed image
6. After a predetermined time, temporary files are cleaned up

## 5. Technology Stack

### 5.1 Frontend Technologies

- **HTML5/CSS3/JavaScript**: Core web technologies
- **React**: Frontend framework for responsive UI
- **WebSockets**: For real-time progress updates
- **Image Comparison Library**: For side-by-side viewing

### 5.2 Backend Technologies

- **Python**: Primary programming language
- **FastAPI/Flask**: Web framework for REST API
- **Redis**: For job queue management
- **JWT**: For secure API access

### 5.3 Machine Learning Stack

- **PyTorch**: Deep learning framework for the deblurring model
- **Pillow/OpenCV**: For image processing operations
- **NumPy**: For efficient numerical operations

### 5.4 Infrastructure

- **Docker**: For containerization
- **Nginx**: Web server and reverse proxy
- **Cloud Storage**: For scalable image storage
- **Cloud Compute**: For hosting the application and model inference

## 6. Deployment Architecture

### 6.1 Deployment Diagram

```
+------------------+      +------------------+
|   Web Server     |      |  API Server      |
|   (Nginx)        |----->|  (Python/FastAPI)|
+------------------+      +------------------+
        |                         |
        |                         |
        |                         v
        |               +------------------+
        |               |  Worker Processes|
        |               |  (CPU-optimized) |
        |               +------------------+
        |                         |
        v                         v
+-----------------------------------------------------------------------------------+
|                           Shared File Storage                                     |
+-----------------------------------------------------------------------------------+
```

### 6.2 Scalability Considerations

- **Horizontal Scaling**: Multiple API servers can be deployed behind a load balancer
- **Worker Pool**: Configure multiple CPU-based worker processes with managed concurrency
- **Queue-Based Architecture**: Decouples upload from processing for better resource utilization
- **Caching**: Implement aggressive caching of results to minimize reprocessing
- **Resource Management**: Careful monitoring and allocation of CPU and memory resources
- **Processing Time Expectations**: Set appropriate user expectations for processing times
- **Job Prioritization**: Implement queue prioritization for smaller/simpler images

### 6.3 Deployment Process

1. Build and test Docker images for each component
2. Deploy containers to cloud infrastructure
3. Configure networking and security
4. Set up monitoring and logging
5. Implement automated scaling based on demand

## 7. Performance Considerations

### 7.1 Model Optimization

- **Model Quantization**: Reduce precision of weights for faster inference
- **Model Pruning**: Remove unnecessary connections to reduce model size
- **CPU Optimization**: Optimize tensor operations for CPU processing
- **Batch Size Management**: Process single images to avoid memory issues
- **Thread Management**: Properly utilize available CPU cores

### 7.2 Application Optimization

- **Image Size Limits**: Enforce reasonable size limits for uploaded images
- **Progressive Loading**: Show results as soon as they're available
- **Background Processing**: Allow users to continue using the application during processing
- **Caching Strategy**: Cache processed images for frequently accessed results

### 7.3 Performance Metrics

- **Processing Time**: Realistic targets of 3-5 minutes for standard images (up to 4MP) on CPU
- **Image Size Limitations**: Strict enforcement of image size limits (recommend max 2-4MP)
- **Concurrent Processing**: Limited to available CPU cores with appropriate memory allocation
- **Concurrent Users**: Support for 50+ simultaneous users with appropriate queue management
- **Upload/Download Speed**: Optimize for various network conditions
- **Server Resource Utilization**: Careful monitoring of CPU, memory, and disk usage
- **Queue Management**: Metrics for queue length and average wait time

## 8. Security Considerations

### 8.1 Data Protection

- **Temporary Storage**: Images stored only for the duration needed
- **Data Isolation**: User data kept separate with proper access controls
- **HTTPS**: All communications encrypted in transit
- **No PII Collection**: Minimal user information collected

### 8.2 API Security

- **Input Validation**: Rigorous validation of all user inputs
- **Rate Limiting**: Prevent abuse through appropriate rate limits
- **CORS Policy**: Strict cross-origin resource sharing configuration
- **Security Headers**: Implementation of best-practice security headers

### 8.3 Infrastructure Security

- **Network Segmentation**: Separate components into appropriate security zones
- **Least Privilege**: Components operate with minimal required permissions
- **Regular Updates**: Timely application of security patches
- **Vulnerability Scanning**: Regular security scans of the application

## 9. Testing Strategy

### 9.1 Unit Testing

- **Component Tests**: Individual tests for each system component
- **Model Tests**: Verification of model behavior under various conditions
- **API Tests**: Validation of API endpoints and responses

### 9.2 Integration Testing

- **Component Integration**: Testing interaction between components
- **End-to-End Flow**: Validation of complete user flows
- **Error Handling**: Verification of system behavior during errors

### 9.3 Performance Testing

- **Load Testing**: System behavior under various load conditions
- **Stress Testing**: System limits and failure modes
- **Scalability Testing**: Validation of scaling mechanisms

### 9.4 User Acceptance Testing

- **Usability Testing**: With representative users
- **Browser Compatibility**: Testing across major browsers
- **Device Compatibility**: Testing on various device types and screen sizes

## 10. Implementation Plan

### 10.1 Phase 1: Core Functionality

- Implement basic web interface for image upload
- Develop strict image preprocessing and validation with size limits
- Optimize PyTorch deblurring model for CPU-only operation
- Implement asynchronous processing with job queue
- Create basic result display with appropriate user expectations for processing time

### 10.2 Phase 2: Enhanced Features

- Implement side-by-side comparison tools
- Add progress indicators
- Optimize model for better performance
- Improve error handling and user feedback

### 10.3 Phase 3: Optimization and Refinement

- Implement caching strategies
- Optimize for mobile devices
- Add analytics for system monitoring
- Polish user interface and experience

## 11. Appendix

### 11.1 Neural Network Details

The deblurring neural network is based on a Generator with dense connections and dilated convolutions. The key parameters include:

- **Input Resolution**: Strictly limited (recommend resizing to 256x256 or 512x512 for CPU processing)
- **Channel Rate**: Consider reducing from 64 to 32 to decrease computational requirements
- **Dense Blocks**: 10 interconnected blocks (consider reducing to 6-8 for CPU optimization)
- **Dilation Factors**: Varying (1, 2, 3, 2, 1) to capture features at different scales
- **Skip Connections**: Both local and global to preserve details
- **Activation Functions**: LeakyReLU (alpha=0.2)
- **Output Activation**: Tanh for normalized output
- **Model Optimization**: Potential for model distillation or pruning to create a lighter version

### 11.2 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/images/upload` | POST | Upload image for processing |
| `/api/jobs/{id}/status` | GET | Check job status |
| `/api/jobs/{id}/result` | GET | Retrieve processing result |
| `/api/health` | GET | System health check |