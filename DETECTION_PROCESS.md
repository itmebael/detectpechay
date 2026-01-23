# AI-Powered Pechay Disease Detection Process

This diagram illustrates how the system processes images to detect Pechay leaf diseases using a Convolutional Neural Network (CNN) via the Roboflow API.

```mermaid
graph TD
    %% Nodes
    Input[📷 Input Image<br/>(Live Stream / Capture)]
    
    subgraph "Local Pre-Processing & Validation"
        Resize[Pre-processing<br/>Resize & Normalize]
        YOLO[YOLOv8 Check<br/>(Person/Object Detection)]
        Quality[Quality Check<br/>(Green Color / Blur)]
    end
    
    subgraph "CNN Analysis (Roboflow API)"
        API_In[API Input Layer]
        Conv1[Convolutional Layer 1<br/>(Edge/Texture Detection)]
        Pool1[Pooling Layer<br/>(Downsampling)]
        Conv2[Convolutional Layer 2<br/>(Shape/Pattern Detection)]
        Deep[Deep Layers<br/>(Complex Feature Isolation)]
        Head[Detection Head<br/>(Classify & Bounding Box)]
    end
    
    Decision{Is Valid?}
    Output[✅ Result Display<br/>(Bounding Box + Label)]
    Block[⛔ Blocked<br/>(Not Pechay / Person)]

    %% Edges
    Input --> Resize
    Resize --> YOLO
    YOLO --> Quality
    Quality --> Decision
    
    Decision -- "Person/Face Detected" --> Block
    Decision -- "No Green / Blur" --> Block
    Decision -- "Valid Leaf" --> API_In
    
    API_In --> Conv1
    Conv1 --> Pool1
    Pool1 --> Conv2
    Conv2 --> Deep
    Deep --> Head
    Head --> Output

    %% Styling
    style Input fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style Output fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style Block fill:#ffebee,stroke:#c62828,stroke-width:2px
    style YOLO fill:#fff3e0,stroke:#ef6c00
    style Head fill:#f3e5f5,stroke:#7b1fa2
```

## Detailed Process Description

### 1. 📷 Input Acquisition
The system captures a frame from the ESP32-CAM live stream.

### 2. 🛡️ Local Validation (The Gatekeeper)
Before sending data to the cloud, the local server performs checks:
*   **YOLOv8 Analysis**: Scans for non-leaf objects (Persons, Cars, etc.).
*   **Color Analysis**: Ensures the image contains sufficient green pixels (characteristic of leaves).
*   **Quality Check**: Rejects blurry or dark images.

### 3. 🧠 CNN Analysis (Roboflow API)
If the image is valid, it is sent to the Roboflow API where the **Convolutional Neural Network (CNN)** operates:
1.  **Feature Extraction**: The network applies multiple filters to the image.
    *   *Early Layers*: Detect simple edges, lines, and color gradients.
    *   *Middle Layers*: Identify shapes like leaf margins, veins, and spot patterns.
    *   *Deep Layers*: Recognize complex disease signatures (e.g., specific lesion textures of Black Rot or Alternaria).
2.  **Localization**: The network determines *where* the disease is located on the leaf (Bounding Box coordinates).
3.  **Classification**: The network assigns a probability (Confidence Score) to the detected region (e.g., "Diseased: 85%").

### 4. ✅ Result Generation
The system draws a bounding box on the dashboard:
*   **Green Box**: Healthy Pechay.
*   **Red Box**: Diseased Pechay (with disease name).
*   **Blocked**: If the local validation failed (e.g., a person was seen).
