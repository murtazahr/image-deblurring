# System Implementation Document

## 1. General Information

### 1.1 Version Control

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| April 08, 2025 | 1.0 | Initial system implementation document | Project Team |

### 1.2 Information Details

| Informational Item | Information |
|-------------------|-------------|
| Project Name | Image Deblurring System |
| Project Phase | Phase 1 |
| Project Iteration | 1 |
| Last Saved On | April 08, 2025 |

## 2. Component Diagram Description

### 2.1 System Components Overview

#### 2.1.1 Component Classification and Definition

| Component | Classification | Definition |
|-----------|----------------|------------|
| Web Frontend | Subsystem | Provides user interface for image upload and result display, implemented as a responsive web application |
| Backend API Service | Subsystem | Manages requests, processes images, and coordinates the deblurring workflow |
| Image Processor | Module | Handles image validation, preprocessing, and post-processing |
| Deblurring Engine | Module | Implements the PyTorch neural network for image deblurring |
| Job Queue Manager | Module | Manages the processing queue and job status |
| Result Storage | Module | Manages temporary storage of input and result images |

#### 2.1.2 Component Responsibilities

| Component | Responsibilities |
|-----------|-----------------|
| Web Frontend | - Display user interface<br>- Handle image file uploads<br>- Present side-by-side comparison<br>- Provide download functionality<br>- Display processing status to users |
| Backend API Service | - Process API requests from frontend<br>- Route images to appropriate processing components<br>- Manage job creation and status<br>- Coordinate the processing workflow |
| Image Processor | - Validate image type, size, and content<br>- Preprocess images (resize, normalize)<br>- Post-process deblurred results<br>- Convert between image formats as needed |
| Deblurring Engine | - Load and manage the PyTorch model<br>- Execute the image deblurring process<br>- Optimize inference performance for CPU processing<br>- Apply model-specific transformations |
| Job Queue Manager | - Create and track processing jobs<br>- Manage job priorities and queue order<br>- Handle timeouts and job cancellation<br>- Provide status updates for ongoing jobs |
| Result Storage | - Store uploaded images temporarily<br>- Manage result image storage<br>- Implement secure access to images<br>- Handle automatic cleanup of expired files |

#### 2.1.3 Component Constraints

| Component | Constraints |
|-----------|------------|
| Web Frontend | - Must support major browsers (Chrome, Firefox, Safari, Edge)<br>- Responsive design for various device sizes<br>- Maximum upload file size: 8MB<br>- Session timeout: 30 minutes |
| Backend API Service | - API rate limits: 20 requests per minute per user<br>- Session handling for security<br>- Graceful degradation under high load |
| Image Processor | - Supported image formats: JPEG, PNG, WebP<br>- Maximum processing resolution: 2048x2048 pixels<br>- Memory constraints: 2GB per process |
| Deblurring Engine | - CPU-only processing (no GPU access)<br>- Memory footprint: Maximum 4GB<br>- Processing time: 3-5 minutes per image<br>- PyTorch model size: ~50MB |
| Job Queue Manager | - Maximum queue length: 100 jobs<br>- Job timeout: 15 minutes<br>- Queue persistence across restarts |
| Result Storage | - Maximum storage capacity: 20GB<br>- File retention period: 24 hours<br>- Secure access via signed URLs |

#### 2.1.4 Component Composition

| Component | Subcomponents |
|-----------|---------------|
| Web Frontend | - Upload Component<br>- Result Viewer Component<br>- Progress Indicator<br>- Image Comparison Tool<br>- Download Manager |
| Backend API Service | - Request Router<br>- Authentication Handler<br>- Response Formatter<br>- Error Handler |
| Image Processor | - Validation Service<br>- Resizing Engine<br>- Normalization Tool<br>- Format Converter |
| Deblurring Engine | - PyTorch Model Wrapper<br>- Generator Neural Network<br>- DenseBlock Implementation<br>- Model Configuration Manager |
| Job Queue Manager | - Queue Processor<br>- Status Tracker<br>- Priority Scheduler<br>- Timeout Monitor |
| Result Storage | - File System Interface<br>- URL Generator<br>- Cleanup Service<br>- Access Control Manager |

#### 2.1.5 Component Interactions

| Component | Uses/Interactions |
|-----------|------------------|
| Web Frontend | - Interacts with Backend API Service to submit jobs and retrieve results<br>- Uses WebSockets for real-time status updates<br>- Presents visual feedback to the user |
| Backend API Service | - Processes requests from Web Frontend<br>- Instructs Image Processor to prepare images<br>- Submits jobs to Job Queue Manager<br>- Retrieves results from Result Storage |
| Image Processor | - Receives images from Backend API Service<br>- Prepares images for Deblurring Engine<br>- Stores processed results in Result Storage |
| Deblurring Engine | - Receives preprocessed images from Image Processor<br>- Returns deblurred results to Image Processor<br>- Manages PyTorch model and inference process |
| Job Queue Manager | - Receives job requests from Backend API Service<br>- Schedules jobs for execution by Deblurring Engine<br>- Updates job status for tracking |
| Result Storage | - Stores files from Image Processor<br>- Provides access to files for Backend API Service<br>- Handles automatic file lifecycle management |

#### 2.1.6 Component Resources

| Component | Resources |
|-----------|-----------|
| Web Frontend | - Static file hosting<br>- CDN for asset delivery<br>- Browser memory and processing |
| Backend API Service | - Application server (e.g., Flask/uWSGI)<br>- Web server (e.g., Nginx)<br>- 2 CPU cores, 4GB RAM |
| Image Processor | - Image processing libraries (Pillow, OpenCV)<br>- 2 CPU cores, 4GB RAM<br>- Temporary storage space |
| Deblurring Engine | - PyTorch runtime<br>- CPU optimized libraries (NumPy, SciPy)<br>- 4 CPU cores, 8GB RAM |
| Job Queue Manager | - Message queue system (e.g., Redis, RabbitMQ)<br>- Persistent storage for job metadata<br>- 1 CPU core, 2GB RAM |
| Result Storage | - Filesystem or object storage<br>- 20GB storage capacity<br>- 1 CPU core, 2GB RAM |

#### 2.1.7 Component Processing

| Component | Processing |
|-----------|-----------|
| Web Frontend | - Client-side validation of image files<br>- Asynchronous file uploads with progress tracking<br>- Dynamic UI updates based on job status<br>- Image comparison visualization using HTML5 Canvas |
| Backend API Service | - REST API implementation with JSON responses<br>- Request validation and sanitization<br>- Error handling with appropriate HTTP status codes<br>- Authentication and session management |
| Image Processor | - Multi-stage pipeline for image preparation<br>- Resize algorithm: Lanczos for downsampling<br>- Color space conversion as needed<br>- Format conversion while preserving metadata |
| Deblurring Engine | - Model loading with weight optimization<br>- Forward pass through generator network<br>- Memory-efficient tensor operations<br>- Sequential processing of image patches for large images |
| Job Queue Manager | - FIFO queue with priority support<br>- Atomic operations for job status updates<br>- Monitoring for stuck or failed jobs<br>- Retry mechanism for transient failures |
| Result Storage | - File deduplication when possible<br>- Content-addressable storage<br>- Signed URL generation for secure access<br>- Time-based expiration and cleanup |

#### 2.1.8 Component Interfaces

| Component | Interface | Description |
|-----------|-----------|-------------|
| Web Frontend | Web UI | Browser-based interface for users to interact with the system |
| Backend API Service | REST API | HTTP endpoints for image upload, job management, and result retrieval |
| Image Processor | ProcessorAPI | Internal API for image validation, preparation, and transformation |
| Deblurring Engine | ModelAPI | Internal API for model loading, inference, and result generation |
| Job Queue Manager | QueueAPI | Internal API for job creation, status updates, and retrieval |
| Result Storage | StorageAPI | Internal API for file storage, retrieval, and lifecycle management |

### 2.2 Interface Specification

#### 2.2.1 REST API Interface

| Interface | REST API |
|-----------|----------|
| Description | HTTP-based API for client-server communication |
| Services | - `POST /api/images/upload`: Upload an image for processing<br>- `GET /api/jobs/{jobId}/status`: Check job status<br>- `GET /api/jobs/{jobId}/result`: Get processing result<br>- `GET /api/health`: Service health check |
| Protocol | - HTTP/HTTPS<br>- JSON for data exchange<br>- JWT for authentication (optional)<br>- Rate limiting applies |
| Notes | Implemented by Backend API Service<br>Use multipart/form-data for image uploads |
| Issues | Need to handle large uploads and timeouts gracefully |

#### 2.2.2 ProcessorAPI Interface

| Interface | ProcessorAPI |
|-----------|----------|
| Description | Internal API for image processing operations |
| Services | - `validate(image)`: Validate image format and size<br>- `preprocess(image, options)`: Prepare image for model<br>- `postprocess(result, options)`: Format result for delivery<br>- `convert(image, target_format)`: Convert between formats |
| Protocol | Function calls with image objects and configuration parameters |
| Notes | Implemented by Image Processor component<br>Uses Pillow and OpenCV internally |
| Issues | Memory usage for large images needs optimization |

#### 2.2.3 ModelAPI Interface

| Interface | ModelAPI |
|-----------|----------|
| Description | Internal API for model inference |
| Services | - `load_model(config)`: Load model with configuration<br>- `predict(image)`: Run inference on preprocessed image<br>- `get_model_info()`: Return model metadata<br>- `optimize_model(level)`: Apply runtime optimizations |
| Protocol | Function calls with tensor objects and configuration parameters |
| Notes | Implemented by Deblurring Engine component<br>Manages PyTorch model lifecycle |
| Issues | Model loading time needs optimization<br>Memory management is critical |

#### 2.2.4 QueueAPI Interface

| Interface | QueueAPI |
|-----------|----------|
| Description | Internal API for job queue management |
| Services | - `create_job(data)`: Create new processing job<br>- `get_job_status(job_id)`: Check job status<br>- `update_job_status(job_id, status)`: Update job status<br>- `get_pending_jobs()`: Retrieve jobs pending processing |
| Protocol | Function calls with job objects and status enumerations |
| Notes | Implemented by Job Queue Manager component<br>Uses atomic operations for consistency |
| Issues | Need to handle queue overflow and job prioritization |

#### 2.2.5 StorageAPI Interface

| Interface | StorageAPI |
|-----------|----------|
| Description | Internal API for file storage |
| Services | - `store_file(data, metadata)`: Store file with metadata<br>- `get_file(file_id)`: Retrieve stored file<br>- `generate_url(file_id, expiry)`: Generate access URL<br>- `delete_file(file_id)`: Remove file from storage |
| Protocol | Function calls with file objects and metadata |
| Notes | Implemented by Result Storage component<br>Handles temporary file storage and management |
| Issues | Automatic cleanup needs to be reliable<br>Security of file access is critical |

## 3. Deployment Diagram Description

### 3.1 Deployment Architecture Overview

The deployment architecture for the Image Deblurring System is designed to be scalable, maintainable, and resource-efficient, considering the CPU-only constraint. The system uses a containerized approach with Docker to ensure consistency across environments and facilitate scaling.

```
+------------------+      +------------------+
|   Web Server     |      |  API Server      |
|   (Nginx)        |----->|  (Python/Flask)  |
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

### 3.2 Deployment Nodes

| Node | Description | Specifications | Software Components |
|------|-------------|---------------|---------------------|
| Web Server | Hosts static files and routes API requests | - 2 CPU cores<br>- 4GB RAM<br>- 20GB storage | - Nginx web server<br>- TLS/SSL termination<br>- Static file hosting<br>- Load balancing |
| API Server | Processes API requests and manages the deblurring pipeline | - 4 CPU cores<br>- 8GB RAM<br>- 40GB storage | - Python Flask application<br>- uWSGI application server<br>- Image Processor component<br>- Job Queue Manager component |
| Worker Processes | Executes the deblurring model on preprocessed images | - 4-8 CPU cores<br>- 16GB RAM<br>- 40GB storage | - PyTorch runtime<br>- Deblurring Engine component<br>- CPU optimization libraries |
| Shared File Storage | Stores uploaded and processed images | - 100GB storage<br>- Redundant storage | - File System or Object Storage<br>- Result Storage component |

### 3.3 Network Connections

| Connection | Description | Protocol | Security |
|------------|-------------|----------|----------|
| User → Web Server | User access to web interface | HTTPS (443) | TLS 1.3, HSTS |
| Web Server → API Server | API request routing | HTTP (8080) | Internal network, API keys |
| API Server → Worker Processes | Job submission and status updates | TCP/Message Queue | Internal network, authentication |
| API Server → Shared Storage | File reading and writing | File System/Object API | Access control, encryption |
| Worker Processes → Shared Storage | Result storage | File System/Object API | Access control, encryption |

### 3.4 Software Distribution

| Component | Container | Base Image | Deployment Configuration |
|-----------|-----------|------------|--------------------------|
| Web Frontend | web-frontend | nginx:alpine | - Port 80, 443<br>- Volume mount for static files<br>- Auto-scaling: 1-3 instances |
| Backend API | backend-api | python:3.10-slim | - Port 8080<br>- Volume mount for temp storage<br>- Auto-scaling: 2-5 instances |
| Worker Processes | deblur-worker | python:3.10-slim | - No exposed ports<br>- Volume mount for model and shared storage<br>- CPU limit: 4 cores per container<br>- Memory limit: 8GB per container<br>- Auto-scaling: 1-4 instances |
| Job Queue | redis-queue | redis:alpine | - Port 6379 (internal)<br>- Persistence volume<br>- Single instance with backup |
| Shared Storage | n/a | n/a | - Mounted across containers<br>- Backup and replication configured |

### 3.5 Deployment Considerations

#### 3.5.1 Scaling Strategy

- **Horizontal Scaling**: The application is designed to scale horizontally with multiple API servers and workers behind a load balancer.
- **Queue-Based Decoupling**: Processing jobs are decoupled from API requests via a message queue, allowing independent scaling of frontend and processing components.
- **Resource-Aware Scaling**: Worker processes are scaled based on CPU utilization, while API servers are scaled based on request volume.
- **Static Content Delivery**: Static frontend assets are cached and delivered through CDN when possible to reduce load on web servers.

#### 3.5.2 Performance Optimizations

- **Model Optimization**: The PyTorch model is optimized for CPU execution using quantization and pruning techniques.
- **Image Size Limitations**: Strict enforcement of image size limits ensures manageable processing times.
- **Memory Management**: Careful memory handling during preprocessing and model inference to avoid Out-of-Memory errors.
- **Batch Processing**: Intelligent batching of smaller images when possible to improve throughput.
- **Caching Strategy**: Caching of commonly used model components and intermediate results.

#### 3.5.3 Failover and Redundancy

- **Container Orchestration**: Kubernetes or Docker Swarm for container management and automatic restart of failed containers.
- **Health Checks**: Regular health checks to detect and recover from component failures.
- **Queue Persistence**: Message queue with persistence to ensure no jobs are lost during restarts.
- **Storage Redundancy**: Redundant storage for uploaded and processed images.
- **Graceful Degradation**: System designed to function with reduced capacity if some components fail.

#### 3.5.4 Monitoring and Operations

- **Logging**: Centralized logging with structured log format for all components.
- **Metrics Collection**: Application and system metrics collection for performance monitoring.
- **Alerting**: Automated alerts for system issues or performance degradation.
- **Dashboard**: Operational dashboard for system status and performance visualization.
- **Deployment Automation**: CI/CD pipeline for automated testing and deployment.

### 3.6 System Success Scenario

1. **User Access**: User accesses the web application via browser and navigates to the upload page.
2. **Image Upload**: User selects and uploads an image for deblurring through the web interface.
3. **Validation**: Web Frontend performs client-side validation, and Backend API performs server-side validation of the uploaded image.
4. **Job Creation**: Backend API creates a processing job and places it in the Job Queue.
5. **Status Tracking**: User is provided with a job ID and status tracking page.
6. **Preprocessing**: Image Processor prepares the image for the deblurring model (resize, normalize).
7. **Job Execution**: Worker Process picks up the job from the queue and executes the deblurring algorithm using the PyTorch model.
8. **Post-processing**: Deblurred image is post-processed to enhance quality and convert to the appropriate format.
9. **Result Storage**: Processed image is stored in Shared Storage with appropriate metadata.
10. **Notification**: User is notified that processing is complete (via UI update if still on the page).
11. **Result Display**: User views the side-by-side comparison of original and deblurred images.
12. **Download**: User downloads the deblurred image to their local device.
13. **Cleanup**: System automatically removes temporary files after 24 hours to free up storage.

This end-to-end flow demonstrates the successful operation of the Image Deblurring System, from user interaction to final result delivery, utilizing all the components described in this document.