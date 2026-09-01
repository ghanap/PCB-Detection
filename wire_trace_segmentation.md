# PCB Wire & Copper Trace Segmentation Engine

Here is the output of the **PCB Wire & Copper Trace Segmentation Engine** (`segment_pcb_wires.py`). 

Instead of filling entire rectangular bounding boxes, this engine extracts the **exact pixel geometry of copper traces, power wires, and ground wires** using lighting-invariant Lab color background subtraction + Otsu thresholding.

![PCB Wire & Copper Trace Mask Preview](C:\Users\ANAGHA\.gemini\antigravity\brain\a18c0ee1-53bb-4527-a3d0-d12dbe11c8f0\pcb_wire_trace_preview.png)

---

## Technical Method: How Trace/Wire Mask Extraction Works

1. **Lab Color Space Conversion**:
   - Converts image crops from RGB to `CIE Lab` color space where Lightness ($L$) is decoupled from color channels ($a, b$).

2. **Border Background Estimation**:
   - Samples pixels along the $3\text{px}$ outer perimeter of the bounding box to calculate the local PCB substrate background color mean $\mu_{bg}$.

3. **Background Subtraction & Otsu Thresholding**:
   - Computes Euclidean distance $d(p) = \|p_{Lab} - \mu_{bg}\|$ for every pixel inside the box.
   - Applies Otsu adaptive thresholding to separate wire/trace foreground from substrate background.

4. **Morphological Contouring**:
   - Applies elliptical morphological closing ($3\times3$) to close solder pad gaps, followed by opening to eliminate small speckle noise.

---

## Deliverable Script

- **PCB Wire & Trace Segmentation Script**: [segment_pcb_wires.py](file:///c:/Userdata/antiiii/segment_pcb_wires.py)
- **Output Mask Directory**: `C:\Userdata\antiiii\wire_trace_masks`
