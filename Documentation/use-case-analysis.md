# Use Case Analysis Document: Image Deblurring Web Tool

## 1. Introduction

### 1.1 Purpose
This document provides a comprehensive use case analysis for the Image Deblurring Web Tool, a final year project that combines a machine learning model for image deblurring with a web-based interface. The purpose of the system is to allow everyday users to easily restore clarity to blurry images without requiring technical expertise in image processing or machine learning.

### 1.2 Scope
This document covers the functionality of both the machine learning model and the web interface components of the system. It includes all user interactions, system processes, and expected outcomes. The document does not cover the detailed implementation of the machine learning algorithms or the specific web technologies used.

### 1.3 Definitions, Acronyms, and Abbreviations

| Term | Definition |
|------|------------|
| ML   | Machine Learning |
| UI   | User Interface |
| UC   | Use Case |
| GAN  | Generative Adversarial Network (likely used in the deblurring model) |
| API  | Application Programming Interface |

### 1.4 References
- PyTorch documentation for image processing models
- Web application framework documentation (specific framework to be determined)
- Image processing research papers (specific references to be added)

### 1.5 Overview
The remainder of this document details the actors involved in the system, provides a comprehensive use case model, and describes each use case in detail, including flows, preconditions, and postconditions.

## 2. Use Case Model Survey

### 2.1 Actors

| Actor | Description |
|-------|-------------|
| End User | A person with limited technical knowledge who wants to deblur images |
| Administrator | A person who manages the web application and monitors system performance |
| ML Model | The machine learning component that processes and deblurs images |
| Storage System | The component that stores uploaded and processed images |

### 2.2 Use Case Diagram

```
+---------------------+
|                     |
|      End User       |
|                     |
+----------+----------+
           |
           |
           v
+----------+----------+     +---------------------+
|                     |     |                     |
|  Upload Blurry Image+---->+   Process Image     |
|                     |     |                     |
+---------------------+     +---------+-----------+
                                      |
                                      |
                                      v
+---------------------+     +---------------------+
|                     |     |                     |
|  Download Result    |<----+  View Results       |
|                     |     |                     |
+---------------------+     +---------------------+

+---------------------+
|                     |
|    Administrator    |
|                     |
+----------+----------+
           |
           |
           v
+----------+----------+
|                     |
| Monitor System Usage|
|                     |
+---------------------+
           |
           |
           v
+----------+----------+
|                     |
| Manage User Content |
|                     |
+---------------------+
```

## 3. Specific Use Cases

### 3.1 Upload Blurry Image

#### 3.1.1 Brief Description
This use case allows an end user to upload a blurry image to the web application for processing.

#### 3.1.2 Actors
- End User
- Storage System

#### 3.1.3 Preconditions
- User has accessed the web application
- User has a blurry image available on their device

#### 3.1.4 Basic Flow
1. User navigates to the upload page
2. System displays the upload interface
3. User selects the "Upload" button
4. System opens a file selection dialog
5. User selects an image file from their device
6. System validates the file (correct format, within size limits)
7. System displays a preview of the uploaded image
8. User confirms the upload
9. System stores the image and assigns it a unique identifier
10. System redirects to the processing page

#### 3.1.5 Alternative Flows

**Alternative Flow 1: Drag and Drop Upload**
1. User navigates to the upload page
2. System displays the upload interface
3. User drags an image file from their device and drops it onto the designated area
4. Continue from step 6 of Basic Flow

#### 3.1.6 Exception Flows

**Exception Flow 1: Invalid File Format**
1. User selects a non-image file or unsupported image format
2. System displays an error message indicating acceptable file formats
3. System returns to the upload interface

**Exception Flow 2: File Size Exceeded**
1. User selects an image file exceeding the maximum allowed size
2. System displays an error message indicating the size limit
3. System returns to the upload interface

#### 3.1.7 Postconditions
- Image is stored in the system with a unique identifier
- User is presented with options to process the image

#### 3.1.8 Special Requirements
- System must support common image formats (JPEG, PNG, etc.)
- Maximum file size limit: 10MB
- Upload interface must be responsive for both desktop and mobile devices

#### 3.1.9 Assumptions
- User has a stable internet connection
- User has permission to upload the image

#### 3.1.10 Dependencies
- None

### 3.2 Process Image

#### 3.2.1 Brief Description
This use case involves the system processing a blurry image using the machine learning model to produce a deblurred version.

#### 3.2.2 Actors
- End User
- ML Model
- Storage System

#### 3.2.3 Preconditions
- User has successfully uploaded an image
- The image is stored in the system

#### 3.2.4 Basic Flow
1. User selects the "Process Image" option
2. System displays processing options (if applicable, such as deblurring intensity)
3. User confirms processing request
4. System queues the image for processing
5. System displays a processing status indicator
6. ML model retrieves the image
7. ML model performs deblurring operation
8. ML model stores the processed image
9. System notifies the user that processing is complete
10. System redirects to the results page

#### 3.2.5 Alternative Flows

**Alternative Flow 1: Batch Processing**
1. User has uploaded multiple images
2. User selects all images to be processed
3. User selects "Process All"
4. System queues all selected images for processing
5. System displays batch processing status
6. ML model processes each image sequentially
7. System updates status as each image completes
8. System notifies user when all processing is complete
9. System redirects to the results gallery

#### 3.2.6 Exception Flows

**Exception Flow 1: Processing Failure**
1. ML model encounters an error during processing
2. System records the error
3. System notifies the user of the failure
4. System offers options to retry or upload a different image

**Exception Flow 2: Server Overload**
1. System detects high processing load
2. System places the request in a longer queue
3. System notifies the user of the extended processing time
4. User can choose to wait or cancel the request

#### 3.2.7 Postconditions
- Deblurred image is stored in the system
- Original and deblurred images are linked in the database

#### 3.2.8 Special Requirements
- ML model must process images in a reasonable time (target: under 60 seconds per image)
- System must handle multiple concurrent processing requests

#### 3.2.9 Assumptions
- ML model has been pre-trained and optimized for a variety of blur types
- Server has adequate computing resources for image processing

#### 3.2.10 Dependencies
- UC 3.1 Upload Blurry Image

### 3.3 View Results

#### 3.3.1 Brief Description
This use case allows the user to view and compare the original blurry image with the deblurred result.

#### 3.3.2 Actors
- End User
- Storage System

#### 3.3.3 Preconditions
- User has submitted an image for processing
- Processing has completed successfully

#### 3.3.4 Basic Flow
1. System displays the results page after processing completes
2. Page shows the original image and the deblurred image side by side
3. User can zoom in on either image to examine details
4. System provides a slider or toggle to switch between views for comparison
5. System displays image quality metrics (optional)
6. User reviews the results

#### 3.3.5 Alternative Flows

**Alternative Flow 1: Gallery View for Multiple Images**
1. User has processed multiple images
2. System displays a gallery of thumbnail pairs (original and processed)
3. User selects a pair to view detailed comparison
4. Continue from step 2 of Basic Flow

#### 3.3.6 Exception Flows

**Exception Flow 1: Image Loading Failure**
1. System cannot retrieve one or both images
2. System displays an error message
3. System offers reload option or return to upload page

#### 3.3.7 Postconditions
- User has viewed the processing results

#### 3.3.8 Special Requirements
- Comparison view must be intuitive and responsive
- Image display must preserve original resolution when zooming

#### 3.3.9 Assumptions
- User's browser supports modern image display techniques
- User has adequate screen resolution to view comparison

#### 3.3.10 Dependencies
- UC 3.2 Process Image

### 3.4 Download Result

#### 3.4.1 Brief Description
This use case allows the user to download the deblurred image to their device.

#### 3.4.2 Actors
- End User
- Storage System

#### 3.4.3 Preconditions
- User has viewed the processing results
- Deblurred image is available in the system

#### 3.4.4 Basic Flow
1. User selects the "Download" option
2. System offers format options (JPEG, PNG, etc.)
3. User selects desired format
4. System prepares the file for download
5. Browser initiates the download to the user's device
6. System confirms successful download

#### 3.4.5 Alternative Flows

**Alternative Flow 1: Download Multiple Images**
1. User has processed multiple images
2. User selects "Download All"
3. System prepares a ZIP archive containing all processed images
4. Browser initiates the download of the archive
5. System confirms successful download

#### 3.4.6 Exception Flows

**Exception Flow 1: Download Failure**
1. Download is interrupted or fails
2. System detects the failure
3. System offers retry option
4. User selects retry
5. System restarts the download process

#### 3.4.7 Postconditions
- Deblurred image is saved on the user's device
- System records the download action (optional)

#### 3.4.8 Special Requirements
- Downloaded images must maintain their quality
- System should offer multiple common image formats

#### 3.4.9 Assumptions
- User has sufficient storage space on their device
- User has permission to save files on their device

#### 3.4.10 Dependencies
- UC 3.3 View Results

### 3.5 Monitor System Usage

#### 3.5.1 Brief Description
This use case allows an administrator to monitor system metrics and usage patterns.

#### 3.5.2 Actors
- Administrator

#### 3.5.3 Preconditions
- Administrator has logged into the admin interface
- Administrator has proper authorization

#### 3.5.4 Basic Flow
1. Administrator navigates to the dashboard
2. System displays key metrics:
    - Number of images processed
    - Average processing time
    - Server load
    - Storage usage
3. Administrator reviews metrics
4. Administrator can filter data by time period
5. Administrator can export metrics reports

#### 3.5.5 Alternative Flows

**Alternative Flow 1: Alert Configuration**
1. Administrator selects "Configure Alerts"
2. System displays alert configuration options
3. Administrator sets thresholds for various metrics
4. Administrator saves configuration
5. System confirms changes

#### 3.5.6 Exception Flows

**Exception Flow 1: Data Retrieval Failure**
1. System cannot retrieve usage data
2. System displays error message
3. Administrator can retry or report the issue

#### 3.5.7 Postconditions
- Administrator has reviewed system usage information

#### 3.5.8 Special Requirements
- Dashboard must update in real-time or near real-time
- System must securely restrict access to admin functions

#### 3.5.9 Assumptions
- Monitoring infrastructure is properly configured
- System collects usage metrics automatically

#### 3.5.10 Dependencies
- None

### 3.6 Manage User Content

#### 3.6.1 Brief Description
This use case allows an administrator to manage user-uploaded content, including review, deletion, and storage management.

#### 3.6.2 Actors
- Administrator
- Storage System

#### 3.6.3 Preconditions
- Administrator has logged into the admin interface
- Administrator has proper authorization

#### 3.6.4 Basic Flow
1. Administrator navigates to content management section
2. System displays a list of user-uploaded and processed images
3. Administrator can filter by date, size, or status
4. Administrator selects content for management
5. Administrator can delete selected content
6. System confirms action before execution
7. System updates storage status after changes

#### 3.6.5 Alternative Flows

**Alternative Flow 1: Automatic Cleanup Configuration**
1. Administrator selects "Configure Auto-Cleanup"
2. System displays configuration options
3. Administrator sets retention policies
4. Administrator saves configuration
5. System confirms changes

#### 3.6.6 Exception Flows

**Exception Flow 1: Delete Failure**
1. System cannot delete selected content
2. System displays error message
3. Administrator can retry or investigate the issue

#### 3.6.7 Postconditions
- Selected content is managed according to administrator actions
- Storage system is updated

#### 3.6.8 Special Requirements
- Bulk operations must be supported
- System must maintain an audit log of admin actions

#### 3.6.9 Assumptions
- Administrator understands content management policies
- Storage system is accessible for management operations

#### 3.6.10 Dependencies
- None

## 4. System Attributes

### 4.1 Performance
- Image upload should complete within 5 seconds for files under 5MB
- Image processing should complete within 60 seconds per image
- Web interface should load within 3 seconds on standard connections
- System should support at least 100 concurrent users

### 4.2 Security
- User data should be protected with appropriate encryption
- Admin access must require strong authentication
- System must comply with relevant data protection regulations
- Images should be stored securely and not accessible without proper authorization

### 4.3 Usability
- Interface should be intuitive for non-technical users
- Mobile responsiveness is required
- Clear feedback must be provided for all user actions
- Help documentation should be easily accessible
- Error messages must be clear and actionable

### 4.4 Reliability
- System should have 99% uptime
- Regular backups of the database should be maintained
- Failover mechanisms should be in place for critical components
- System should gracefully handle unexpected inputs or errors

## 5. Appendices

### 5.1 User Interface Mockups

**Home Page / Upload Screen**
```
+------------------------------------------+
|  Image Deblurring Tool                [?]|
+------------------------------------------+
|                                          |
|  +----------------------------------+    |
|  |                                  |    |
|  |                                  |    |
|  |       Drag & Drop Images         |    |
|  |               or                 |    |
|  |       [Select Files to Upload]   |    |
|  |                                  |    |
|  +----------------------------------+    |
|                                          |
|  Supported formats: JPG, PNG, TIFF       |
|  Maximum file size: 10MB                 |
|                                          |
+------------------------------------------+
```

**Results Comparison Screen**
```
+------------------------------------------+
|  Image Deblurring Tool             [🏠] |
+------------------------------------------+
|  Original             Deblurred          |
|  +---------------+   +---------------+   |
|  |               |   |               |   |
|  |               |   |               |   |
|  |               |   |               |   |
|  |               |   |               |   |
|  +---------------+   +---------------+   |
|                                          |
|  [Download]  [Compare]  [Try Another]    |
|                                          |
+------------------------------------------+
```

### 5.2 Data Models

**Image Entity**
```
- image_id: UUID (Primary Key)
- original_filename: String
- upload_timestamp: DateTime
- file_size: Integer
- file_format: String
- status: Enum (Uploaded, Processing, Completed, Failed)
- original_image_path: String
- processed_image_path: String (nullable)
- processing_time: Integer (nullable)
- user_session_id: String
```

**Processing Job**
```
- job_id: UUID (Primary Key)
- image_id: UUID (Foreign Key)
- start_time: DateTime
- completion_time: DateTime (nullable)
- status: Enum (Queued, Processing, Completed, Failed)
- error_message: String (nullable)
- processing_parameters: JSON (nullable)
```

### 5.3 Additional Information

**Deblurring Model Information**
The machine learning model for image deblurring will be based on a Generative Adversarial Network (GAN) architecture, specifically adapted for image restoration tasks. The model will be trained on a diverse dataset of blurry and clear image pairs to learn the mapping from blurred to sharp images. The implementation will use PyTorch and will be optimized for both performance and quality.

**Future Extensions**
Potential future extensions to the system may include:
- Specialized deblurring for specific image types (portraits, landscapes, documents)
- Batch processing capabilities with email notifications
- User accounts for saving processing history
- API access for integration with other applications
- Mobile application for direct capture and processing