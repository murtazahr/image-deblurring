# System Testing Document

## 1. General Information

### 1.1 Version Control

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| April 11, 2025 | 1.0 | Initial system testing document | Project Team |

### 1.2 Information Details

| Informational Item | Information |
|-------------------|-------------|
| Document Title | Image Deblurring System Testing Document |
| Version | 1.0 |
| Author | Project Team |
| Project Name | Image Deblurring Web Tool |
| Project Phase | Phase 1 |
| Project Iteration | 1 |
| Last Saved On | April 11, 2025 |
| Number of Pages | 12 |

## 2. Testing Process

The testing process for the Image Deblurring System follows standard software testing practices with specific adaptations for machine learning components. Our testing strategy incorporates both traditional software testing approaches for the web interface and specialized validation techniques for the neural network model.

The process includes:

1. Unit testing of individual components
2. Integration testing of connected components
3. System testing of the complete application
4. Performance testing under various conditions
5. Usability testing with representative users
6. Model-specific testing for deblurring quality

Test results are documented in test logs and incident reports, with a final summary report compiled at the end of each testing cycle.

## 3. Definitions in Testing Process

- **Acceptance criteria**: The criteria that the Image Deblurring System must satisfy to be accepted by users and project stakeholders.
- **Deblurring quality**: Measurement of how effectively the system removes blur from images while preserving important details.
- **Processing time**: The time taken to deblur an image from upload completion to result availability.
- **User journey**: The complete path a user takes when interacting with the system, from upload to download.
- **Pass/Fail criteria**: Decision rules used to determine whether a component or feature passes or fails a test.
- **Test case**: A set of conditions or variables under which a tester will determine whether the system satisfies requirements.
- **Test procedure**: Detailed instructions for the setup, execution, and evaluation of results for a given test case.

## 4. Test Plan

### 4.1 Test Plan Identifier

IDST-TP-2025-001

### 4.2 Introduction

This test plan covers the Image Deblurring System, which consists of a PyTorch-based neural network with dense field architecture for image deblurring and a web-based user interface. The system allows users to upload blurry images, process them through the deblurring model, and download the restored results.

References:
- Project authorization: PA-2025-042
- Project plan: PP-IDS-2025-001

### 4.3 Test Items

The following items will be tested:
- Web frontend interface (HTML/CSS/JavaScript)
- Backend API service (Python/Flask)
- Image preprocessing module
- PyTorch deblurring model
- Result display and comparison view
- Image download functionality

Version: 1.0 for all components

### 4.4 Features to be Tested

1. **Image Upload Functionality**
   - File selection and upload
   - Drag and drop functionality
   - Progress indication
   - File validation (format, size)

2. **Image Preprocessing**
   - Image validation
   - Resizing for model input
   - Normalization

3. **Deblurring Processing**
   - Queue management
   - Processing status updates
   - Error handling during processing

4. **Results Presentation**
   - Side-by-side comparison view
   - Zoom and pan functionality
   - Before/after toggle

5. **Image Download**
   - Download processed image
   - Format selection

6. **Core Model Performance**
   - Deblurring effectiveness on various blur types
   - Processing time optimization
   - Error handling

### 4.5 Features Not to be Tested

1. User account management (not implemented in Phase 1)
2. Batch processing of multiple images (planned for future phases)
3. Mobile application integration (outside current scope)
4. Third-party storage integration (outside current scope)
5. Advanced customization options for deblurring parameters

### 4.6 Approach

The testing approach combines automated and manual testing methods:

1. **Frontend Testing**
   - Manual testing of UI components for functionality and usability
   - Cross-browser testing on Chrome, Firefox, and Safari
   - Responsive design testing on desktop and tablet viewports

2. **Backend API Testing**
   - Automated tests for API endpoints using pytest
   - Load testing with simulated concurrent users
   - Error handling validation

3. **Model Testing**
   - Quantitative assessment using standard image quality metrics (PSNR, SSIM)
   - Visual inspection of deblurring results
   - Performance benchmarking on standard hardware

4. **Integration Testing**
   - End-to-end user journey testing
   - File handling between components
   - Error propagation between components

Minimum degree of comprehensiveness: Basic functional testing of all major components, with focus on complete user journey coverage for critical paths. Unit testing will be applied selectively to core functionality rather than targeting specific coverage percentages.

Testing will be constrained by:
- Limited to CPU-based processing (no GPU acceleration)
- Maximum image size of 2048x2048 pixels
- 4.5-month timeline for development and testing

### 4.7 Item Pass/Fail Criteria

**Web Interface Components:**
- All critical user journeys can be completed without errors
- UI renders correctly on target browsers
- Response time under 3 seconds for non-processing operations

**Backend API:**
- All endpoints return expected responses for valid inputs
- All endpoints return appropriate error codes for invalid inputs
- Can handle at least 10 concurrent users

**Image Processing:**
- Successfully processes images up to 2048x2048 pixels
- Correctly identifies and rejects unsupported file formats
- Preserves aspect ratio during resizing

**Deblurring Model:**
- PSNR improvement of at least 3dB for test images
- SSIM improvement of at least 0.1 for test images
- Processing time under 5 minutes for 1024x1024 pixel images on target hardware
- No significant artifacts introduced in the deblurred image

### 4.8 Suspension Criteria and Resumption Requirements

Testing will be suspended if:
- Critical defects that prevent basic functionality are discovered
- Backend services become unavailable
- Test environment becomes unstable

Testing will resume when:
- Critical defects have been addressed
- Services are restored and stable
- Environment issues are resolved

All tests affected by the suspension must be restarted from the beginning upon resumption.

### 4.9 Test Deliverables

[Section removed as per project requirements]

### 4.10 Testing Tasks

[Section removed as per project requirements]

### 4.11 Environmental Needs

**Hardware:**
- Development laptops/desktops with minimum 16GB RAM, quad-core processors
- Test server with 8-core CPU, 32GB RAM for model testing
- Network connectivity for distributed testing

**Software:**
- Python 3.10+ with PyTorch 2.0+
- Flask development server
- Modern web browsers (latest versions of Chrome, Firefox, Safari)
- Testing frameworks: pytest, Selenium
- Image processing libraries: Pillow, OpenCV

**Other:**
- Test dataset of blurry images (various types and sizes)
- Benchmark dataset with ground truth pairs (blurry and sharp)
- Network bandwidth sufficient for image upload/download testing

## 5. Test Case Specification

### 5.1 Test Case Specification Identifier

IDST-TC-2025-001

### 5.2 Test Items

This section contains specifications for key test cases. Each test case focuses on a specific functionality or feature of the Image Deblurring System.

### 5.3 Input Specifications

#### Test Case ID: TC-UPLOAD-001
**Description:** Verify that users can upload images through the web interface
**Inputs:**
- Valid JPEG image file (1024x768, 500KB)
- Valid PNG image file (800x600, 1.2MB)
- Invalid file format (PDF, 100KB)
- Oversized image (JPEG, 20MB)

#### Test Case ID: TC-PROCESS-001
**Description:** Verify that system processes images correctly
**Inputs:**
- Motion-blurred image (1024x768, medium blur)
- Defocus-blurred image (1024x768, medium blur)
- Combined blur image (motion + defocus)

#### Test Case ID: TC-RESULT-001
**Description:** Verify correct display of results comparison
**Inputs:**
- Processed image pair (original + deblurred)

#### Test Case ID: TC-DOWNLOAD-001
**Description:** Verify download functionality for processed images
**Inputs:**
- Deblurred image available for download

### 5.4 Output Specifications

#### Test Case ID: TC-UPLOAD-001
**Expected Outputs:**
- Valid images: Upload confirmation, preview displayed
- Invalid format: Error message indicating supported formats
- Oversized: Error message indicating size limit

#### Test Case ID: TC-PROCESS-001
**Expected Outputs:**
- Processing status updates
- Successful completion notification
- Deblurred image with improved clarity
- PSNR improvement: 3-5 dB
- SSIM improvement: 0.1-0.3

#### Test Case ID: TC-RESULT-001
**Expected Outputs:**
- Side-by-side comparison displayed
- Zooming functionality working
- Toggle between views functioning

#### Test Case ID: TC-DOWNLOAD-001
**Expected Outputs:**
- Download initiated
- File saved with correct format and content
- File name preserved with appropriate suffix

### 5.5 Environmental Needs

Tests will be executed on:
- Hardware: Standard development machine (16GB RAM, quad-core CPU)
- Software: Latest Chrome browser, Flask development server
- Network: Standard broadband connection

### 5.6 Special Procedural Requirements

- TC-PROCESS-001 requires monitoring of system resources during execution
- TC-UPLOAD-001 requires clean environment state before each execution

### 5.7 Intercase Dependencies

- TC-RESULT-001 depends on successful completion of TC-PROCESS-001
- TC-DOWNLOAD-001 depends on successful completion of TC-RESULT-001

## 6. Test Procedure Specification

### 6.1 Test Procedure Specification Identifier

IDST-TP-2025-001

### 6.2 Purpose

This procedure specifies the steps to execute the test cases for the Image Deblurring System, ensuring comprehensive validation of all key features.

### 6.3 Test Procedure

#### 6.3.1 Start
1. Initialize the test environment
2. Start the backend server
3. Launch the web application in the browser
4. Prepare test images for upload

#### 6.3.2 Proceed
1. Execute test cases in the following order:
   - TC-UPLOAD-001
   - TC-PROCESS-001
   - TC-RESULT-001
   - TC-DOWNLOAD-001
2. Record observations and results for each step
3. Document any unexpected behavior

#### 6.3.3 Measure
- Record upload time for various image sizes
- Measure processing time for deblurring operation
- Calculate PSNR and SSIM between original and deblurred images
- Monitor system resource usage during processing
- Record user interface response times

#### 6.3.4 Shut Down
1. Close all browser instances
2. Shut down backend server
3. Archive test results

#### 6.3.5 Restart
In case of interruption:
1. Restore environment to clean state
2. Restart backend server
3. Resume testing from the last completed test case

#### 6.3.6 Stop
1. Complete all test cases
2. Ensure all results are recorded
3. Shut down all components

#### 6.3.7 Wrap Up
1. Consolidate test results
2. Generate metrics and reports
3. Clean up test environment
4. Archive test artifacts

#### 6.3.8 Contingencies
- If backend crashes: record error logs, restart server, and resume testing
- If browser crashes: restart browser and continue from last step
- If deblurring fails: record error conditions and investigate

### 6.4 Procedure Results

The following results were observed during execution of the test procedures:

**TC-UPLOAD-001:**
- Valid JPEG and PNG images were uploaded successfully
- Preview displayed correctly
- Invalid file format was rejected with appropriate error message
- Oversized image was rejected with size limit notification
- Result: PASS

**TC-PROCESS-001:**
- Processing status updates displayed correctly
- Motion-blurred image: Successfully processed, PSNR improvement: 4.2 dB, SSIM improvement: 0.18
- Defocus-blurred image: Successfully processed, PSNR improvement: 3.8 dB, SSIM improvement: 0.22
- Combined blur image: Successfully processed, PSNR improvement: 3.5 dB, SSIM improvement: 0.15
- Average processing time: 3 minutes 42 seconds (1024x768 image)
- Result: PASS

**TC-RESULT-001:**
- Side-by-side comparison displayed correctly
- Zoom and pan functionality working as expected
- Toggle between views functioning correctly
- High-detail regions show noticeable improvement in clarity
- Result: PASS

**TC-DOWNLOAD-001:**
- Download initiated successfully
- File saved with correct format and content
- Filename preserved with "_deblurred" suffix
- Result: PASS

### 6.5 Anomalous Events

1. Intermittent delay observed in status updates during processing of larger images
   - Occurred when processing images larger than 1600x1200 pixels
   - Updates would pause for 10-15 seconds before resuming
   - Functionality not affected, only UI responsiveness

2. Occasional slower processing time for images with complex textures
   - Observed 20-30% increase in processing time for certain landscape images
   - Output quality not affected

### 6.6 Variances

1. Processing time variance:
   - Specification: Under 5 minutes for 1024x1024 pixel images
   - Actual: 2-4 minutes depending on image content and blur type
   - Reason: Variation in image complexity and blur characteristics

2. UI design variance:
   - Specification: Progress bar for upload and processing
   - Actual: Implemented progress percentage text for upload and animated icon for processing
   - Reason: Simplified implementation for Phase 1, maintaining equivalent functionality

### 6.7 Summary of Results

The Image Deblurring System testing has yielded positive results across all major functionality areas:

**Image Upload and Validation:**
- Successfully handles supported image formats (JPEG, PNG)
- Correctly validates file format and size
- Provides appropriate error messages for invalid files
- Upload interface is intuitive and responsive

**Image Processing:**
- Deblurring model effectively improves image clarity across different blur types
- Average PSNR improvement: 3.8 dB (exceeding the 3 dB requirement)
- Average SSIM improvement: 0.18 (exceeding the 0.1 requirement)
- Processing time within acceptable range (avg. 3min 42sec for 1024x768 images)
- Processing status updates provide adequate user feedback

**Results Presentation:**
- Side-by-side comparison effectively demonstrates improvements
- Visualization tools (zoom, pan, toggle) function correctly
- UI is responsive and intuitive

**Image Download:**
- Download functionality works reliably
- Output files maintain quality and expected format

**Resolved Incidents:**
1. Fixed UI freezing during large file uploads by implementing chunked uploading
2. Resolved memory leak in preprocessing module that occurred with certain image dimensions
3. Fixed incorrect EXIF data handling that caused orientation issues in some photos
4. Addressed browser compatibility issues in the comparison view for Safari

**Conclusion:**
The Image Deblurring System meets all critical requirements and passes the defined acceptance criteria. The system demonstrates effective deblurring capabilities with acceptable processing times on standard hardware. The web interface provides an intuitive user experience for uploading, processing, and downloading images.

The identified minor issues do not impact core functionality and can be addressed in subsequent iterations. The system is ready for user acceptance testing and preliminary release within the academic context.

**Recommendations:**
1. Implement optimizations for images with complex textures
2. Refine status update mechanism for more consistent feedback
3. Consider progressive loading of results for larger images
4. Explore CPU optimization techniques to further reduce processing time