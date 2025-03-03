# Automated Access Control Policy Component Annotation with ACPUIE and doccano

## Overview
This guide provides steps to deploy ACPUIE (fine-tuned UIE) and configure doccano for automated annotation of access control policies.

---

## Step 1: Fine-tune and Test ACPUIE

1. ​**Deploy UIE**​  
   Follow the instructions in [PaddleNLP/slm/model_zoo/uie/README.md](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/slm/model_zoo/uie/README.md) to set up UIE.

2. ​**Prepare Data**​  
   Place the provided `python` folder under the `uie` directory. Use a small subset from our dataset for fine-tuning.

3. ​**Test ACPUIE**​  
   Run the following command to verify functionality:
   ```bash
   python test.py
## Step 2: Deploy doccano with Auto-Annotation

1. ​**Set Up doccano**​  
   Follow the official guide at [doccano/doccano](https://github.com/doccano/doccano) to deploy the annotation tool.

2. ​**Launch Custom API**​  
   Start the custom REST API:
   ```bash
   python api.py
3. **Configure Auto-Labeling**

    a. Navigate to ​**​Settings → Auto Labeling → Custom REST Request**.

    b. Set the URL to the IP and port used by `api.py`.

    c. In the ​**Body**​ section, add:  
   ```json
   { 
     "key":"text"
     "text": "{{ text }}"
   }
   ```
    d. Add the following **​Mapping Template**:
    ```bash
    [
        {% for entity in input %}
            {
                "start_offset": {{ entity.start_offset }},
                "end_offset": {{ entity.end_offset}},
                "label": "{{ entity.label }}"
            }{% if not loop.last %},{% endif %}
        {% endfor %}
    ]
    ```
    e. Ensure doccano labels exactly match the names of Access Policy Components.
