# Feasibility Study Report

## 1. Introduction

### 1.1 Purpose

The purpose of this feasibility study is to evaluate the technical, operational, and economic viability of developing an Image Deblurring System as an undergraduate final year project. This study aims to identify potential challenges, assess resource requirements, and determine the overall feasibility of successfully completing the project within the given constraints.

### 1.2 Background

Digital images are frequently compromised by blur, which can result from various factors including camera motion, defocus, or subject movement during capture. This degradation significantly impacts image quality and can render important visual information unusable for both personal and professional applications. Current solutions often require specialized technical knowledge or expensive software, creating barriers for everyday users who need to recover details from blurred photos.

Our proposed solution consists of a web-based application that incorporates a neural network model based on a dense field architecture to effectively remove blur from images while preserving important details. This tool aims to make sophisticated deblurring technology accessible to everyday users without requiring technical expertise.

### 1.3 Methodology

The feasibility assessment was conducted through the following methods:

1. Review of available technologies and frameworks for web application development
2. Analysis of the PyTorch model architecture and CPU processing requirements
3. Evaluation of development tools and environments
4. Assessment of team skills and knowledge gaps
5. Estimation of development timelines and resource requirements
6. Identification of potential risks and mitigation strategies

### 1.4 References

1. Project Synopsis document
2. PyTorch model implementation code
3. Software Project Plan
4. Software Requirements Specification
5. Sequence Analysis and State Diagram documents
6. Course materials on software development methodologies

## 2. General Information

### 2.1 Current Systems and Processes

Currently, users with blurred images have the following options:

1. **Professional Software**: Tools like Adobe Photoshop that offer deblurring features but require expensive licenses and technical expertise.
2. **Mobile Apps**: Various mobile applications that provide basic deblurring but often with limited effectiveness and quality.
3. **Online Services**: Web-based tools that may offer simple deblurring features but typically with significant limitations on image size, quality, or require payment.
4. **Manual Processing**: Seeking help from professionals with image processing expertise, which can be expensive and time-consuming.

For most everyday users, none of these options provide an accessible and effective solution for recovering information from blurred images. This creates a clear opportunity for a user-friendly web application that leverages advanced machine learning techniques.

### 2.2 System Objectives

The Image Deblurring System aims to achieve the following objectives:

1. **Accessibility**: Create a web-based interface that can be used by individuals without technical expertise.
2. **Effectiveness**: Implement a deep learning model capable of significantly improving image clarity for various types of blur.
3. **Usability**: Develop an intuitive user experience with features like side-by-side comparison of original and deblurred images.
4. **Performance**: Optimize the system to deliver reasonable processing times on standard hardware without GPU acceleration.
5. **Educational Value**: Demonstrate the practical application of deep learning for image processing and web application development as part of an undergraduate final year project.

### 2.3 Issues

Several issues need to be addressed for successful implementation:

1. **Computational Requirements**: The neural network model is computationally intensive, especially without GPU acceleration. Processing times may be longer than ideal for a web-based application.

2. **Model Optimization**: The model will need to be optimized for CPU-only processing, which may involve trade-offs between processing speed and deblurring quality.

3. **Scalability**: With limited resources, the system may face challenges handling multiple simultaneous users.

4. **Image Size Limitations**: Constraints on memory and processing power will necessitate limits on the size of images that can be processed.

5. **Technical Knowledge Gap**: As an undergraduate project, there may be learning curves associated with implementing advanced deep learning models and web development.

### 2.4 Assumptions and Constraints

The following assumptions and constraints guide this feasibility assessment:

1. **Development Team**: Two undergraduate students with basic knowledge of machine learning and web development.

2. **Timeline**: Approximately 4.5 months for complete development and testing.

3. **Hardware Limitations**: No access to GPU acceleration; all processing will be CPU-based.

4. **Hosting Environment**: Will utilize free tier cloud resources for deployment.

5. **Scope Limitations**: The project will focus on a core set of features and exclude advanced functionality such as user accounts, batch processing, or custom deblurring parameters.

6. **Academic Context**: The project must satisfy academic requirements while still delivering a functional application.

## 3. Alternatives

Three alternative approaches have been considered for this project:

### 3.1 Alternative 1: Full-stack Web Application with PyTorch Model

**Description**: Develop a complete web application with a frontend interface, backend API, and integrated PyTorch model for image deblurring.

**Components**:
- React or simple HTML/CSS/JavaScript frontend
- Python Flask/FastAPI backend
- PyTorch model optimized for CPU processing
- Simple job queue for processing management
- Basic file storage for images

**Pros**:
- Provides a complete end-to-end solution
- Offers valuable learning experience across the full stack
- Demonstrates practical application of the deblurring model
- Accessible to end users through a web browser

**Cons**:
- More complex to implement within the time constraints
- Requires knowledge across multiple domains
- Performance limitations without GPU acceleration

### 3.2 Alternative 2: Desktop Application with PyTorch Model

**Description**: Create a standalone desktop application that incorporates the PyTorch model for local image processing.

**Components**:
- Python-based GUI (e.g., Tkinter, PyQt)
- Integrated PyTorch model
- Local file system for image storage

**Pros**:
- Eliminates web hosting concerns
- Potentially better performance with direct access to local resources
- Simpler architecture with fewer components
- Easier to implement security (no network exposure)

**Cons**:
- Limited accessibility (requires installation)
- Platform-specific development challenges
- Less relevant to modern cloud-based application development
- Limited portability across devices

### 3.3 Alternative 3: Simplified Web UI with Pre-trained Model API

**Description**: Create a basic web interface that connects to an existing third-party image processing API for deblurring.

**Components**:
- Simple web frontend
- Integration with third-party API for image processing
- Minimal backend for API proxying

**Pros**:
- Significantly reduced development complexity
- No need to handle resource-intensive processing
- Faster development timeline
- Focus on UI/UX aspects

**Cons**:
- Reduced learning opportunity in deep learning implementation
- Dependency on third-party services
- Potential costs for API usage
- Limited control over the deblurring algorithm and results
- Less fulfilling as a capstone project

### 3.4 Comparison of Alternatives

| Criteria | Alt 1: Web App with PyTorch | Alt 2: Desktop App | Alt 3: Web UI with 3rd Party API |
|----------|------------------------------|--------------------|---------------------------------|
| **Technical Feasibility** | Medium | High | Very High |
| **Learning Value** | High | Medium | Low |
| **User Accessibility** | High | Low | High |
| **Development Complexity** | High | Medium | Low |
| **Resource Requirements** | Medium | Low | Very Low |
| **Control Over Implementation** | High | High | Low |
| **Alignment with Project Goals** | High | Medium | Low |
| **Scalability** | Medium | N/A | High |
| **Maintenance Requirements** | Medium | Low | Very Low |

## 4. Recommendations and Conclusions

### 4.1 Recommended Approach

Based on the analysis of alternatives, we recommend pursuing **Alternative 1: Full-stack Web Application with PyTorch Model** with the following modifications to ensure feasibility:

1. **Simplified Scope**: Focus on core functionality with strict limitations on image size and processing parameters.

2. **Optimized Model**: Implement the proposed model architecture with optimizations for CPU processing, including:
    - Reduced channel count (from 64 to 32)
    - Lower resolution processing (max 512x512 pixels)
    - Potential reduction in dense blocks if necessary for performance

3. **Asynchronous Processing**: Implement a simple job queue system to handle processing in the background, allowing users to retrieve results when ready.

4. **Clear User Expectations**: Set appropriate expectations for processing times (3-5 minutes) in the user interface.

5. **Progressive Implementation**: Develop the system in phases, starting with core functionality and adding enhancements as time permits.

### 4.2 Implementation Plan

1. **Phase 1 (Weeks 1-4)**:
    - Set up development environment
    - Optimize the PyTorch model for CPU processing
    - Create basic backend API structure
    - Implement image validation and preprocessing

2. **Phase 2 (Weeks 5-10)**:
    - Develop frontend interface with upload functionality
    - Implement processing job queue
    - Create result storage and retrieval system
    - Integrate model with backend

3. **Phase 3 (Weeks 11-16)**:
    - Implement side-by-side comparison view
    - Add progress indicators and status tracking
    - Optimize image processing pipeline
    - Create download functionality

4. **Phase 4 (Weeks 17-18)**:
    - Testing and bug fixing
    - Documentation completion
    - Deployment to cloud hosting
    - Final presentation preparation

### 4.3 Risk Assessment

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| Processing time too slow for user acceptance | High | Medium | Set clear expectations, optimize model, implement progress indicators |
| Memory issues with large images | Medium | High | Implement strict image size limitations, process in patches if necessary |
| Learning curve steeper than anticipated | Medium | Medium | Allocate time for learning, prioritize core functionality, seek guidance from advisors |
| Integration issues between components | Medium | Medium | Use proven frameworks, implement incremental integration, thorough testing |
| Deployment challenges | Low | Medium | Research hosting options early, develop with deployment in mind |

### 4.4 Conclusion

The development of an Image Deblurring System as an undergraduate final year project is feasible with the recommended approach and careful management of constraints. The project offers significant educational value while addressing a practical need for accessible image deblurring tools.

By focusing on a carefully scoped web application with an optimized implementation of the dense field neural network model, the two-person team can successfully complete the project within the 4.5-month timeframe. The main challenges will be optimizing the model for CPU-only processing and managing user expectations regarding processing times.

We conclude that this project is technically viable, educationally valuable, and can be completed within the given constraints, making it a suitable choice for an undergraduate final year project.